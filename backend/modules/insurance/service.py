import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
from typing import Tuple, Dict, List
from core.database import load_ehr_data
from supabase import Client
from config.settings import get_settings
from modules.insurance.schemas import RatingFactor, AgeBand, PlanSimulation
from supabase import create_client

settings = get_settings()
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
MODEL_PATH = "ai_models/insurance_glm.pkl"

CHRONIC_ICDS = ['E11', 'I50', 'J44', 'I25', 'C50', 'C34', 'C18']

def load_and_prepare() -> pd.DataFrame:
    df = load_ehr_data()

    # Keep only needed columns
    core_cols = [
        'patient_id', 'claim_id', 'enroll_patient_gender', 'enroll_patient_year_of_birth',
        'enroll_patient_state', 'enroll_benefit_type', 'enroll_pay_type',
        'enroll_date_start', 'enroll_date_end', 'med_date_service', 'med_date_service_end',
        'med_location_of_care', 'diag_diagnosis_code', 'total_paid_cost', 'total_patient_cost',
        'total_submitted_cost', 'pharm_ndc'
    ]
    df = df[[c for c in core_cols if c in df.columns]].copy()

    # Parse dates
    date_cols = ['med_date_service', 'med_date_service_end', 'enroll_date_start', 'enroll_date_end']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Numeric
    for col in ['enroll_patient_year_of_birth', 'total_paid_cost', 'total_patient_cost', 'total_submitted_cost']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['age_at_service'] = df['med_date_service'].dt.year - df['enroll_patient_year_of_birth']
    df['tenure_days'] = (df['med_date_service'] - df['enroll_date_start']).dt.days
    df['is_active'] = (df['med_date_service'] >= df['enroll_date_start']) & (df['med_date_service'] <= df['enroll_date_end'])
    df['claim_intensity'] = df['total_submitted_cost'] / (df['total_paid_cost'] + 1e-6)
    df['has_hospital'] = df['med_location_of_care'].isin(['Inpatient', 'Hospital'])
    df['has_er'] = df['med_location_of_care'] == 'Emergency Room'
    df['has_pharmacy'] = df['pharm_ndc'].notna()
    df['diag_prefix'] = df['diag_diagnosis_code'].astype(str).str[:3]
    df['has_chronic'] = df['diag_prefix'].isin(CHRONIC_ICDS)
    return df

def aggregate_member_level(df: pd.DataFrame) -> pd.DataFrame:
    member_df = df.groupby('patient_id').agg(
        age=('age_at_service', 'mean'),
        gender=('enroll_patient_gender', 'first'),
        state=('enroll_patient_state', 'first'),
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

    member_df['member_months'] = member_df['tenure_days'] / 30.44
    member_df['pmpm_allowed'] = member_df['total_allowed'] / member_df['member_months'].replace(0, np.nan)
    return member_df.dropna(subset=['pmpm_allowed'])

def train_model(member_df: pd.DataFrame):
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

    X_test_const = sm.add_constant(X_test, has_constant='add')
    y_pred = result.predict(X_test_const)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model
    os.makedirs("ai_models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((result, X_train_const.columns), f)

    return result, list(X_train_const.columns), mae, r2

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def generate_rating_factors(model, cols) -> Tuple[List[RatingFactor], float]:
    coef = model.params
    base_rate = np.exp(coef.get('const', 0))
    factors = []
    for feat in coef.index:
        if feat == 'const':
            continue
        factors.append(RatingFactor(
            feature=feat,
            coefficient=float(coef[feat]),
            factor=np.exp(coef[feat])
        ))
    return sorted(factors, key=lambda x: x.factor, reverse=True)[:10], base_rate

def generate_age_table(model, cols, base_rate) -> List[AgeBand]:
    ages = list(range(0, 81, 5))
    rows = []
    for age in ages:
        row = pd.Series(0, index=cols)
        row['const'] = 1
        row['age'] = age
        pred = model.predict(row.to_frame().T)[0]
        rows.append(AgeBand(age=age, predicted_pmpm=pred, relative_to_base=pred/base_rate))
    return rows

def simulate_plan(member_df: pd.DataFrame, ded: float, coin: float, oop_max: float, name: str) -> PlanSimulation:
    sim = member_df.copy()
    sim['oop_sim'] = np.minimum(sim['total_allowed'] * coin + ded, oop_max)
    sim['plan_pays'] = sim['total_allowed'] - sim['oop_sim']
    return PlanSimulation(
        plan=name,
        deductible=ded,
        coinsurance=coin,
        oop_max=oop_max,
        avg_oop_per_member=sim['oop_sim'].mean(),
        avg_plan_cost_pmpm=sim['plan_pays'].sum() / sim['member_months'].sum(),
        total_plan_cost=sim['plan_pays'].sum()
    )

def run_insurance_analysis() -> Dict:
    df = load_and_prepare()
    df = engineer_features(df)
    member_df = aggregate_member_level(df)

    model, cols = load_model()
    if not model:
        model, cols, mae, r2 = train_model(member_df)
    else:
        mae, r2 = 0.0, 0.0  # Skip eval

    factors, base_rate = generate_rating_factors(model, cols)
    age_table = generate_age_table(model, cols, base_rate)

    plans = [
        ("Rich", 0, 0.1, 2000),
        ("Standard", 500, 0.2, 4000),
        ("HDHP", 1500, 0.2, 6000)
    ]
    simulations = [simulate_plan(member_df, d, c, m, n) for n, d, c, m in plans]

    result = {
        "base_rate": base_rate,
        "mae": mae,
        "r2": r2,
        "top_rating_factors": [f.dict() for f in factors],
        "age_rating_table": [a.dict() for a in age_table],
        "sample_plans": [s.dict() for s in simulations]
    }

    supabase.table("insurance_model_results").upsert(result).execute()
    return result