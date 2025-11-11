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
FILE_NAME_2 = "combined_output_brotli_1.parquet"
response1 = supabase.storage.from_(supabase_bucket).download(FILE_NAME_1)
response2 = supabase.storage.from_(supabase_bucket).download(FILE_NAME_2)
df1 = pd.read_parquet(io.BytesIO(response1))
df2 = pd.read_parquet(io.BytesIO(response2))

# Merge dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Ensure numeric columns
df['total_submitted_cost'] = pd.to_numeric(df['total_submitted_cost'], errors='coerce')
df['total_paid_cost'] = pd.to_numeric(df['total_paid_cost'], errors='coerce')
df['total_patient_cost'] = pd.to_numeric(df['total_patient_cost'], errors='coerce')

def validate_claim_fields(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['claim_id', 'total_submitted_cost', 'total_paid_cost', 'total_patient_cost', 'proc_procedure_code']
    invalid_claims = df[df[required_cols].isnull().any(axis=1)]
    result = invalid_claims[['claim_id', 'total_submitted_cost', 'total_paid_cost', 'total_patient_cost', 'proc_procedure_code', 'med_date_service', 'med_date_service_end', 'proc_date_service']].copy()
    result['med_date_service'] = pd.to_datetime(result['med_date_service'], errors='coerce').dt.strftime('%Y-%m-%d')
    result['med_date_service_end'] = pd.to_datetime(result['med_date_service_end'], errors='coerce').dt.strftime('%Y-%m-%d')
    result['proc_date_service'] = pd.to_datetime(result['proc_date_service'], errors='coerce').dt.strftime('%Y-%m-%d')
    return result

def detect_workflow_inefficiencies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = ['claim_id', 'total_submitted_cost', 'total_paid_cost', 'total_patient_cost', 'proc_procedure_code', 'med_date_service', 'med_date_service_end', 'proc_date_service']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    duplicate_claims = df[df.duplicated(['claim_id', 'proc_procedure_code', 'proc_date_service'], keep=False)]
    df['med_date_service'] = pd.to_datetime(df['med_date_service'], errors='coerce')
    df['med_date_service_end'] = pd.to_datetime(df['med_date_service_end'], errors='coerce')
    df['turnaround_days'] = (df['med_date_service_end'] - df['med_date_service']).dt.days
    long_turnaround_claims = df[df['turnaround_days'] > 1]
    
    duplicate_result = duplicate_claims[required_cols].copy()
    long_turnaround_result = long_turnaround_claims[required_cols].copy()
    
    duplicate_result['med_date_service'] = pd.to_datetime(duplicate_result['med_date_service'], errors='coerce').dt.strftime('%Y-%m-%d')
    duplicate_result['med_date_service_end'] = pd.to_datetime(duplicate_result['med_date_service_end'], errors='coerce').dt.strftime('%Y-%m-%d')
    duplicate_result['proc_date_service'] = pd.to_datetime(duplicate_result['proc_date_service'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    long_turnaround_result['med_date_service'] = pd.to_datetime(long_turnaround_result['med_date_service'], errors='coerce').dt.strftime('%Y-%m-%d')
    long_turnaround_result['med_date_service_end'] = pd.to_datetime(long_turnaround_result['med_date_service_end'], errors='coerce').dt.strftime('%Y-%m-%d')
    long_turnaround_result['proc_date_service'] = pd.to_datetime(long_turnaround_result['proc_date_service'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    return (duplicate_result, long_turnaround_result)

# Process data
invalid_claims = validate_claim_fields(df)
duplicate_claims, long_turnaround_claims = detect_workflow_inefficiencies(df)

# Convert to JSON-serializable format
invalid_claims_list = invalid_claims.fillna('').to_dict(orient='records')
duplicate_claims_list = duplicate_claims.fillna('').to_dict(orient='records')
long_turnaround_claims_list = long_turnaround_claims.fillna('').to_dict(orient='records')

# Upload to Supabase
supabase.table("claims_validation").upsert({
    "invalid_claims": invalid_claims_list,
    "duplicate_claims": duplicate_claims_list,
    "long_turnaround_claims": long_turnaround_claims_list
}).execute()

print("Data uploaded to Supabase table 'claims_validation'")