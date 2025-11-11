from datetime import date
from typing import Optional, List
import pandas as pd
from core.database import load_ehr_data

def get_patient_data(
    patient_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    diagnosis_contains: Optional[str] = None,
    procedure_contains: Optional[str] = None,
    benefit_type: Optional[str] = None
) -> pd.DataFrame:
    df = load_ehr_data()

    # Group by patient + date
    df_grouped = df.groupby(['patient_id', 'med_date_service']).agg({
        'diag_diagnosis_code': lambda x: ', '.join(sorted(set(str(i) for i in x.dropna()))),
        'proc_procedure_code': lambda x: ', '.join(sorted(set(str(i) for i in x.dropna()))),
        'total_submitted_cost': 'sum',
        'total_paid_cost': 'sum',
        'enroll_benefit_type': lambda x: ', '.join(sorted(set(str(i) for i in x.dropna())))
    }).reset_index()

    patient_df = df_grouped[df_grouped['patient_id'] == patient_id].copy()

    if start_date:
        patient_df = patient_df[patient_df['med_date_service'].dt.date >= start_date]
    if end_date:
        patient_df = patient_df[patient_df['med_date_service'].dt.date <= end_date]
    if diagnosis_contains:
        patient_df = patient_df[patient_df['diag_diagnosis_code'].str.contains(diagnosis_contains, case=False, na=False)]
    if procedure_contains:
        patient_df = patient_df[patient_df['proc_procedure_code'].str.contains(procedure_contains, case=False, na=False)]
    if benefit_type:
        patient_df = patient_df[patient_df['enroll_benefit_type'].str.contains(benefit_type, case=False, na=False)]

    return patient_df.sort_values('med_date_service')