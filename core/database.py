# core/database.py
import pandas as pd
import io
from supabase import create_client
from config.settings import get_settings
from functools import lru_cache

settings = get_settings()
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

FILE_NAMES = [
    "combined_output_brotli_1.parquet",
    "combined_output_brotli_2.parquet"
]

# ALL COLUMNS NEEDED BY ANY MODULE
REQUIRED_COLUMNS = {
    # Patient Module
    'patient_id', 'med_date_service', 'diag_diagnosis_code',
    'proc_procedure_code', 'total_submitted_cost', 'total_paid_cost',
    'enroll_benefit_type',

    # Operations Module
    'enroll_date_start', 'enroll_date_end',
    'diag_date_service', 'proc_date_service',

    # Billing Module
    'claim_id', 'total_patient_cost',
    'med_date_service_end',

    # Insurance Module
    'enroll_patient_gender', 'enroll_patient_year_of_birth',
    'enroll_patient_zip3', 'enroll_patient_state',
    'enroll_pay_type', 'med_location_of_care',
    'pharm_date_service', 'pharm_ndc',
    'proc_procedure_units', 'proc_line_charge', 'proc_line_allowed',
    'prov_npi', 'prov_taxonomy_code', 'total_claim_difference'
}

@lru_cache
def load_ehr_data() -> pd.DataFrame:
    dfs = []
    for fname in FILE_NAMES:
        try:
            data = supabase.storage.from_(settings.BUCKET_NAME).download(fname)
            df_part = pd.read_parquet(io.BytesIO(data))
            dfs.append(df_part)
        except Exception as e:
            print(f"Warning: Failed to load {fname}: {e}")

    if not dfs:
        raise ValueError("No data loaded from any file.")

    df = pd.concat(dfs, ignore_index=True)

    # Keep only required + available columns
    available_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[available_cols].copy()

    # Convert date columns
    date_columns = {
        'med_date_service', 'med_date_service_end', 'enroll_date_start',
        'enroll_date_end', 'diag_date_service', 'proc_date_service',
        'pharm_date_service'
    }
    for col in date_columns & set(df.columns):
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert numeric columns
    numeric_columns = {
        'total_submitted_cost', 'total_paid_cost', 'total_patient_cost',
        'enroll_patient_year_of_birth', 'proc_procedure_units',
        'proc_line_charge', 'proc_line_allowed', 'total_claim_difference'
    }
    for col in numeric_columns & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing patient_id
    if 'patient_id' in df.columns:
        df.dropna(subset=['patient_id'], inplace=True)

    print(f"Loaded EHR data: {len(df):,} rows, {len(df.columns)} columns")
    return df