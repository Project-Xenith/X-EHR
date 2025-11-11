import argparse
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore") 
from dotenv import load_dotenv
import io
from supabase import create_client
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_bucket = os.getenv("BUCKET_NAME")

supabase = create_client(supabase_url, supabase_key)
FILE_NAME_1 = "combined_output_brotli_2.parquet"
FILE_NAME_2 = "combined_output_brotli_1.parquet"
response1 = supabase.storage.from_(supabase_bucket).download(FILE_NAME_1)
response2 = supabase.storage.from_(supabase_bucket).download(FILE_NAME_2)
df1 = pd.read_parquet(io.BytesIO(response1))
df2 = pd.read_parquet(io.BytesIO(response2))

# Merge dataframes
df = pd.concat([df1, df2], ignore_index=True)
CORE_COLUMNS = [
    'claim_id', 'patient_id',
    'enroll_patient_gender', 'enroll_patient_year_of_birth',
    'enroll_patient_zip3', 'enroll_patient_state',
    'enroll_date_start', 'enroll_date_end',
    'enroll_benefit_type', 'enroll_pay_type',
    'med_date_service', 'med_date_service_end',
    'med_location_of_care', 'med_pay_type',
    'diag_diagnosis_code',
    'proc_date_service', 'proc_procedure_code', 'proc_procedure_units',
    'proc_line_charge', 'proc_line_allowed',
    'pharm_date_service', 'pharm_ndc', 'pharm_days_supply', 'pharm_dispensed_quantity',
    'prov_npi', 'prov_taxonomy_code',
    'total_submitted_cost', 'total_paid_cost', 'total_patient_cost',
    'total_claim_difference'
]

CHRONIC_ICDS = ['E11', 'I50', 'J44', 'I25', 'C50', 'C34', 'C18']

def load_and_clean_data(filepath: str) -> pd.DataFrame:

    df = df[CORE_COLUMNS].copy()

    # Parse dates
    date_cols = ['med_date_service', 'med_date_service_end', 'enroll_date_start',
                 'enroll_date_end', 'proc_date_service', 'pharm_date_service']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert year of birth
    df['enroll_patient_year_of_birth'] = pd.to_numeric(df['enroll_patient_year_of_birth'], errors='coerce')

    # Convert cost fields
    cost_cols = ['total_paid_cost', 'total_patient_cost', 'total_submitted_cost']
    for c in cost_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all member-level features."""
    print("Engineering features...")

    # Age at service
    df['age_at_service'] = df['med_date_service'].dt.year - df['enroll_patient_year_of_birth']

    # Tenure
    df['tenure_days'] = (df['med_date_service'] - df['enroll_date_start']).dt.days

    # Active membership
    df['is_active'] = (df['med_date_service'] >= df['enroll_date_start']) & \
                      (df['med_date_service'] <= df['enroll_date_end'])

    # Cost metrics
    df['allowed_per_member'] = df['total_paid_cost']
    df['oop_per_member'] = df['total_patient_cost']
    df['claim_intensity'] = df['total_submitted_cost'] / (df['total_paid_cost'] + 1e-6)

    # Utilization flags
    df['has_hospital'] = df['med_location_of_care'].isin(['Inpatient', 'Hospital'])
    df['has_er'] = df['med_location_of_care'] == 'Emergency Room'
    df['has_pharmacy'] = df['pharm_ndc'].notna()

    # Chronic condition
    df['diag_prefix'] = df['diag_diagnosis_code'].astype(str).str[:3]
    df['has_chronic'] = df['diag_prefix'].isin(CHRONIC_ICDS)

    return df
def aggregate_to_member_level(df: pd.DataFrame) -> pd.DataFrame:
    """Roll up claim-level data to member-level."""
    print("Aggregating to member level...")
    member_df = df.groupby('patient_id').agg(
        age=('age_at_service', 'mean'),
        gender=('enroll_patient_gender', 'first'),
        state=('enroll_patient_state', 'first'),
        zip3=('enroll_patient_zip3', 'first'),
        benefit_type=('enroll_benefit_type', 'first'),
        pay_type=('enroll_pay_type', 'first'),
        tenure_days=('tenure_days', 'mean'),
        total_allowed=('total_paid_cost', 'sum'),
        total_oop=('total_patient_cost', 'sum'),
        claim_count=('claim_id', 'nunique'),
        has_hospital=('has_hospital', 'max'),
        has_er=('has_er', 'max'),
        has_pharmacy=('has_pharmacy', 'max'),
        has_chronic=('has_chronic', 'max'),
        avg_claim_intensity=('claim_intensity', 'mean')
    ).reset_index()

    # PMPM
    member_df['member_months'] = member_df['tenure_days'] / 30.44
    member_df['pmpm_allowed'] = member_df['total_allowed'] / member_df['member_months'].replace(0, np.nan)
    member_df = member_df.dropna(subset=['pmpm_allowed'])

    print(f"Member-level records: {len(member_df)}")
    return member_df
def train_glm_model(member_df: pd.DataFrame):
    """Train Gamma GLM and return model + data."""
    print("Training GLM model...")
    cat_features = ['gender', 'state', 'benefit_type', 'pay_type']
    num_features = ['age', 'tenure_days', 'claim_count', 'has_hospital', 'has_er',
                    'has_pharmacy', 'has_chronic', 'avg_claim_intensity']

    model_data = member_df[cat_features + num_features + ['pmpm_allowed']].copy()
    model_data = model_data.apply(pd.to_numeric, errors='coerce')
    model_data = pd.get_dummies(model_data, columns=cat_features, drop_first=True)

    X = model_data.drop('pmpm_allowed', axis=1).astype(float)
    y = model_data['pmpm_allowed'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_const = sm.add_constant(X_train)
    glm = sm.GLM(y_train, X_train_const, family=sm.families.Gamma(link=sm.families.links.log()))
    result = glm.fit()

    # Evaluation
    X_test_const = sm.add_constant(X_test, has_constant='add')
    y_pred = result.predict(X_test_const)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model trained. MAE: ${mae:.2f}, R²: {r2:.3f}")
    return result, X_train_const.columns, X_test_const, y_test, y_pred
def generate_rating_factors(model_result, model_cols) -> pd.DataFrame:
    coef = model_result.params
    base_rate = np.exp(coef['const'])
    factors = pd.DataFrame({
        'feature': coef.drop('const').index,
        'coefficient': coef.drop('const').values,
        'factor': np.exp(coef.drop('const'))
    })
    return factors, base_rate


def generate_age_rating_table(model_result, model_cols, base_rate) -> pd.DataFrame:
    age_bands = pd.DataFrame({'age': range(0, 81, 5)})
    for col in model_cols:
        if col == 'age':
            continue
        age_bands[col] = 1 if col == 'const' else 0
    if 'gender_Male' in age_bands.columns:
        age_bands['gender_Male'] = 1
    age_bands = age_bands[model_cols]
    age_bands['predicted_pmpm'] = model_result.predict(age_bands)
    age_bands['relative_to_base'] = age_bands['predicted_pmpm'] / base_rate
    return age_bands[['age', 'predicted_pmpm', 'relative_to_base']]

def simulate_benefit(member_df: pd.DataFrame, deductible=0, coinsurance=0.2, oop_max=5000):
    sim = member_df.copy()
    sim['oop_sim'] = np.minimum(sim['total_allowed'] * coinsurance + deductible, oop_max)
    sim['plan_pays'] = sim['total_allowed'] - sim['oop_sim']
    return {
        'avg_oop_per_member': sim['oop_sim'].mean(),
        'avg_plan_cost_pmpm': sim['plan_pays'].sum() / sim['member_months'].sum(),
        'total_plan_cost': sim['plan_pays'].sum()
    }

def main():
    parser = argparse.ArgumentParser(description="Policy Design Pipeline")
    parser.add_argument('--input', type=str, required=True, help='Path to input file (CSV/Parquet)')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1–3: Load → Clean → Engineer → Aggregate
    df = load_and_clean_data(args.input)
    df = engineer_features(df)
    member_df = aggregate_to_member_level(df)

    # 4: Train model
    model_result, model_cols, X_test_const, y_test, y_pred = train_glm_model(member_df)

    # 5: Rating factors
    rating_factors, base_rate = generate_rating_factors(model_result, model_cols)
    age_table = generate_age_rating_table(model_result, model_cols, base_rate)

    # 6: Plan simulation
    plans = [("Rich", 0, 0.1, 2000), ("Standard", 500, 0.2, 4000), ("HDHP", 1500, 0.2, 6000)]
    results = []
    for name, ded, coin, maxoop in plans:
        res = simulate_benefit(member_df, ded, coin, maxoop)
        res['plan'] = name
        results.append(res)
    plan_comparison = pd.DataFrame(results)
    print(f"\nAll done! Results saved to {args.output}")
    print("\nTop 5 Rating Factors:")
    print(rating_factors.sort_values('factor', ascending=False).head(5))
    print("\nSample Age Table:")
    print(age_table.head(3))
    print("\nPlan Comparison:")
    print(plan_comparison[['plan', 'avg_plan_cost_pmpm']])

if __name__ == "__main__":
    main()