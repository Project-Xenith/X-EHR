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

@lru_cache
def load_ehr_data() -> pd.DataFrame:
    settings = get_settings()
    dfs = []
    for fname in FILE_NAMES:
        data = supabase.storage.from_(settings.BUCKET_NAME).download(fname)
        df = pd.read_parquet(io.BytesIO(data))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    relevant_cols = [
        'patient_id', 'med_date_service', 'diag_diagnosis_code',
        'proc_procedure_code', 'total_submitted_cost', 'total_paid_cost',
        'enroll_benefit_type'
    ]
    df_clean = df[relevant_cols].copy()
    df_clean.dropna(subset=['patient_id', 'med_date_service'], inplace=True)

    df_clean['med_date_service'] = pd.to_datetime(df_clean['med_date_service'], errors='coerce')
    df_clean['total_submitted_cost'] = pd.to_numeric(df_clean['total_submitted_cost'], errors='coerce')
    df_clean['total_paid_cost'] = pd.to_numeric(df_clean['total_paid_cost'], errors='coerce')
    df_clean.dropna(subset=['med_date_service'], inplace=True)

    return df_clean