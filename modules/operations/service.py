# modules/operations/service.py
from datetime import date
from typing import Dict, List, Tuple
import pandas as pd
from core.database import load_ehr_data
from supabase import Client
from config.settings import get_settings
from supabase import create_client

settings = get_settings()
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def get_operational_insights(
    quarter_start: date,
    quarter_end: date,
    top_n: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns filtered DataFrame + insights dict.
    """
    df = load_ehr_data()

    # Ensure date columns exist and are datetime
    date_cols = ["enroll_date_start", "enroll_date_end", "diag_date_service", "proc_date_service"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Drop rows where critical dates are missing
    df = df.dropna(subset=["enroll_date_start", "enroll_date_end"])

    q_start = pd.to_datetime(quarter_start)
    q_end = pd.to_datetime(quarter_end)

    # Active patients: enrollment overlaps with quarter
    active_patients = (
        (df["enroll_date_start"] <= q_end) &
        (df["enroll_date_end"] >= q_start)
    )
    active_df = df[active_patients]

    # Diagnoses & Procedures within quarter
    diag_df = active_df[
        active_df["diag_date_service"].between(q_start, q_end)
    ] if "diag_date_service" in active_df.columns else pd.DataFrame()

    proc_df = active_df[
        active_df["proc_date_service"].between(q_start, q_end)
    ] if "proc_date_service" in active_df.columns else pd.DataFrame()

    top_diagnoses = (
        diag_df["diag_diagnosis_code"].value_counts().head(top_n).to_dict()
        if not diag_df.empty else {}
    )
    top_procedures = (
        proc_df["proc_procedure_code"].value_counts().head(top_n).to_dict()
        if not proc_df.empty else {}
    )

    insights = {
        "quarter": f"{quarter_start} to {quarter_end}",
        "total_active_patients": int(active_df["patient_id"].nunique()),
        "top_diagnoses": top_diagnoses,
        "top_procedures": top_procedures,
    }

    return active_df, insights


def save_insights_to_supabase(insights: Dict):
    """Upsert operational insights into Supabase."""
    data = {
        "quarter": insights["quarter"],
        "top_diagnoses": insights["top_diagnoses"],
        "top_procedures": insights["top_procedures"],
        "total_active_patients": insights["total_active_patients"],
    }
    supabase.table("operational_insights").upsert(data).execute()