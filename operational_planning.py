import pandas as pd
import io
from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv() 

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_bucket = os.getenv("BUCKET_NAME")
FILE_NAME = "combined_output_brotli_2.parquet"

supabase = create_client(supabase_url, supabase_key)
FILE_NAME = "combined_output_brotli_2.parquet"
response = supabase.storage.from_(supabase_bucket).download(FILE_NAME)
df = pd.read_parquet(io.BytesIO(response))

import pandas as pd

def hospital_operational_insights(df, quarter_start, quarter_end, top_n=5):
    # Convert dates
    df["enroll_date_start"] = pd.to_datetime(df["enroll_date_start"])
    df["enroll_date_end"] = pd.to_datetime(df["enroll_date_end"])
    df["diag_date_service"] = pd.to_datetime(df["diag_date_service"])
    df["proc_date_service"] = pd.to_datetime(df["proc_date_service"])

    # Convert quarter boundaries
    q_start = pd.to_datetime(quarter_start)
    q_end = pd.to_datetime(quarter_end)

    # Filter for patients active during the quarter
    active_df = df[(df["enroll_date_start"] <= q_end) & (df["enroll_date_end"] >= q_start)]

    # Filter diagnoses and procedures within the quarter
    diag_df = active_df[(active_df["diag_date_service"] >= q_start) & (active_df["diag_date_service"] <= q_end)]
    proc_df = active_df[(active_df["proc_date_service"] >= q_start) & (active_df["proc_date_service"] <= q_end)]

    # Count top diagnosis codes
    top_diagnoses = diag_df["diag_diagnosis_code"].value_counts().head(top_n)

    # Count top procedure codes
    top_procedures = proc_df["proc_procedure_code"].value_counts().head(top_n)

    # Format result
    result = {
        "Quarter": f"{quarter_start} to {quarter_end}",
        "Top Diagnoses": top_diagnoses.to_dict(),
        "Top Procedures": top_procedures.to_dict(),
        "Total Active Patients": active_df["patient_id"].nunique()
    }

    return result

insights = hospital_operational_insights(df, "2023-01-01", "2025-03-31")
supabase.table("operational_insights").upsert({
    "quarter": insights["Quarter"],
    "top_diagnoses": insights["Top Diagnoses"],
    "top_procedures": insights["Top Procedures"],
    "total_active_patients": insights["Total Active Patients"]
}).execute()
