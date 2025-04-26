import pandas as pd
import io
from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv() 

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_bucket = os.getenv("BUCKET_NAME")

supabase = create_client(supabase_url, supabase_key)
FILE_NAME_1 = "combined_output_brotli_2.parquet"
response = supabase.storage.from_(supabase_bucket).download(FILE_NAME_1)
FILE_NAME_2 = "combined_output_brotli_1.parquet"
df1 = pd.read_parquet(io.BytesIO(response))
df2 = pd.read_parquet(io.BytesIO(supabase.storage.from_(supabase_bucket).download(FILE_NAME_2)))

#merge 2 dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Ensure numeric columns
df['total_submitted_cost'] = pd.to_numeric(df['total_submitted_cost'], errors='coerce')
df['total_paid_cost'] = pd.to_numeric(df['total_paid_cost'], errors='coerce')
df['total_patient_cost'] = pd.to_numeric(df['total_patient_cost'], errors='coerce')

def transparent_cost_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    cost_breakdown = df.groupby('claim_id').agg({
        'total_submitted_cost': 'sum',
        'total_paid_cost': 'sum',
        'total_patient_cost': 'sum'
    }).reset_index()

    cost_breakdown['unpaid_cost'] = (
        cost_breakdown['total_submitted_cost'] -
        cost_breakdown['total_paid_cost'] -
        cost_breakdown['total_patient_cost']
    )
    
    print("🔍 Transparent Cost Breakdown (sample):")
    print(cost_breakdown.head())
    return cost_breakdown

def validate_claim_fields(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['claim_id', 'total_submitted_cost', 'total_paid_cost', 'total_patient_cost', 'proc_procedure_code']
    invalid_claims = df[df[required_cols].isnull().any(axis=1)]

    print("⚠️ Claims with Missing/Inconsistent Data (sample):")
    print(invalid_claims[['claim_id', 'proc_procedure_code', 'total_submitted_cost']].drop_duplicates().head())
    return invalid_claims


def detect_workflow_inefficiencies(df: pd.DataFrame) -> pd.DataFrame:
    # Duplicates
    duplicate_claims = df[df.duplicated(['claim_id', 'proc_procedure_code', 'proc_date_service'], keep=False)]

    # Turnaround
    df['med_date_service'] = pd.to_datetime(df['med_date_service'])
    df['med_date_service_end'] = pd.to_datetime(df['med_date_service_end'])
    df['turnaround_days'] = (df['med_date_service_end'] - df['med_date_service']).dt.days

    long_turnaround_claims = df[df['turnaround_days'] > 1]

    print("📌 Duplicate Claims Detected (sample):")
    print(duplicate_claims[['claim_id', 'proc_procedure_code', 'proc_date_service']].drop_duplicates().head())

    print("\n📊 Claims with Long Turnaround Time (sample):")
    print(long_turnaround_claims[['claim_id', 'turnaround_days']].drop_duplicates().head())

    return pd.concat([duplicate_claims, long_turnaround_claims]).drop_duplicates()

# costs = transparent_cost_breakdown(df)
# invalids = validate_claim_fields(df)
# inefficiencies = detect_workflow_inefficiencies(df)

