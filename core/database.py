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

# All columns that any module might need
REQUIRED_COLS = {
    # Patient module
    'patient_id', 'med_date_service', 'diag_diagnosis_code',
    'proc_procedure_code', 'total_submitted_cost', 'total_paid_cost',
    'enroll_benefit_type',
    # Operations module
    'enroll_date_start', 'enroll_date_end',
    'diag_date_service', 'proc_date_service',
}

@lru_cache
def load_ehr_data() -> pd.DataFrame:
    dfs = []
    for fname in FILE_NAMES:
        data = supabase.storage.from_(settings.BUCKET_NAME).download(fname)
        df_part = pd.read_parquet(io.BytesIO(data))
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    # Keep only required columns (silently ignore missing ones)
    available_cols = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[available_cols].copy()

    # Convert date columns (coerce errors → NaT)
    date_columns = {
        'med_date_service', 'enroll_date_start', 'enroll_date_end',
        'diag_date_service', 'proc_date_service'
    }
    for col in date_columns & set(df.columns):
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert cost columns
    cost_columns = {'total_submitted_cost', 'total_paid_cost'}
    for col in cost_columns & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing critical identifiers
    df.dropna(subset=['patient_id'], inplace=True)

    return df