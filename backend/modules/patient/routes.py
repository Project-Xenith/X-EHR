# modules/patient/routes.py
from fastapi import APIRouter, HTTPException, Query
from datetime import date
from typing import Optional
from modules.patient.service import get_patient_data
from modules.patient.schemas import PatientSummaryResponse, Record
from ai_insights.patient_summarizer import generate_medical_summary
import pandas as pd

router = APIRouter(prefix="/patient", tags=["Patient"])

@router.get("/{patient_id}/summary", response_model=PatientSummaryResponse)
async def get_patient_summary(
    patient_id: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    diagnosis: Optional[str] = Query(None),
    procedure: Optional[str] = Query(None),
    benefit_type: Optional[str] = Query(None)
):
    df = get_patient_data(
        patient_id=patient_id,
        start_date=start_date,
        end_date=end_date,
        diagnosis_contains=diagnosis,
        procedure_contains=procedure,
        benefit_type=benefit_type
    )

    if df.empty:
        raise HTTPException(404, "No records found for this patient.")

    records = [
        {
            "date": row['med_date_service'].date(),
            "diag": row['diag_diagnosis_code'],
            "proc": row['proc_procedure_code']
        }
        for _, row in df.iterrows()
    ]
    medical_summary = generate_medical_summary(records)

    response = PatientSummaryResponse(
        patient_id=patient_id,
        medical_summary=medical_summary,
        total_submitted=float(df['total_submitted_cost'].sum()),
        total_paid=float(df['total_paid_cost'].sum()),
        record_count=len(df),
        records=[
            Record(
                date=row['med_date_service'].date(),
                diagnosis_codes=row['diag_diagnosis_code'],
                procedure_codes=row['proc_procedure_code'],
                submitted_cost=float(row['total_submitted_cost']),
                paid_cost=float(row['total_paid_cost']),
                benefit_type=row['enroll_benefit_type']
            )
            for _, row in df.iterrows()
        ]
    )
    return response