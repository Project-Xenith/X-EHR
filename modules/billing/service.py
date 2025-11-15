# modules/billing/service.py
from typing import List, Tuple, Dict
import pandas as pd
from core.database import load_ehr_data
from supabase import Client
from config.settings import get_settings
from modules.billing.schemas import ClaimRecord
from supabase import create_client

settings = get_settings()
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

REQUIRED_COLS = [
    'claim_id', 'total_submitted_cost', 'total_paid_cost',
    'total_patient_cost', 'proc_procedure_code',
    'med_date_service', 'med_date_service_end', 'proc_date_service'
]

def validate_claim_fields(df: pd.DataFrame) -> List[Dict]:
    available = [c for c in REQUIRED_COLS if c in df.columns]
    if not available:
        return []

    missing_mask = df[available].isnull().any(axis=1)
    invalid_df = df[missing_mask].copy()

    # Format dates
    for col in ['med_date_service', 'med_date_service_end', 'proc_date_service']:
        if col in invalid_df.columns:
            invalid_df[col] = pd.to_datetime(invalid_df[col], errors='coerce').dt.date

    return invalid_df[available].fillna('').to_dict(orient='records')

def detect_workflow_inefficiencies(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    dup_key = ['claim_id', 'proc_procedure_code', 'proc_date_service']
    dup_available = all(c in df.columns for c in dup_key)

    # Duplicates
    duplicate_records = []
    if dup_available:
        dup_df = df[df.duplicated(dup_key, keep=False)].copy()
        for col in ['med_date_service', 'med_date_service_end', 'proc_date_service']:
            if col in dup_df.columns:
                dup_df[col] = pd.to_datetime(dup_df[col], errors='coerce').dt.date
        duplicate_records = dup_df[dup_key + ['total_submitted_cost', 'total_paid_cost']].fillna('').to_dict(orient='records')

    # Long turnaround
    long_turnaround_records = []
    if all(c in df.columns for c in ['med_date_service', 'med_date_service_end']):
        df['med_date_service'] = pd.to_datetime(df['med_date_service'], errors='coerce')
        df['med_date_service_end'] = pd.to_datetime(df['med_date_service_end'], errors='coerce')
        df['turnaround_days'] = (df['med_date_service_end'] - df['med_date_service']).dt.days
        long_df = df[df['turnaround_days'] > 1].copy()
        for col in ['med_date_service', 'med_date_service_end', 'proc_date_service']:
            if col in long_df.columns:
                long_df[col] = long_df[col].dt.date
        long_df['turnaround_days'] = long_df['turnaround_days'].astype('Int64')
        long_turnaround_records = long_df[[
            'claim_id', 'total_submitted_cost', 'total_paid_cost', 'total_patient_cost',
            'proc_procedure_code', 'med_date_service', 'med_date_service_end',
            'proc_date_service', 'turnaround_days'
        ]].fillna('').to_dict(orient='records')

    return duplicate_records, long_turnaround_records

def run_billing_analysis() -> Dict:
    df = load_ehr_data()

    # Ensure numeric
    for col in ['total_submitted_cost', 'total_paid_cost', 'total_patient_cost']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    invalid_claims = validate_claim_fields(df)
    duplicate_claims, long_turnaround_claims = detect_workflow_inefficiencies(df)

    result = {
        "invalid_claims": invalid_claims,
        "duplicate_claims": duplicate_claims,
        "long_turnaround_claims": long_turnaround_claims,
        "total_invalid": len(invalid_claims),
        "total_duplicates": len(duplicate_claims),
        "total_long_turnaround": len(long_turnaround_claims),
    }

    return result

def save_billing_results_to_supabase(result: Dict):
    data = {
        "invalid_claims": result["invalid_claims"],
        "duplicate_claims": result["duplicate_claims"],
        "long_turnaround_claims": result["long_turnaround_claims"],
        "summary": {
            "total_invalid": result["total_invalid"],
            "total_duplicates": result["total_duplicates"],
            "total_long_turnaround": result["total_long_turnaround"],
        }
    }
    supabase.table("claims_validation").upsert(data).execute()