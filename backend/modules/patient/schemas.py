# modules/patient/schemas.py
from pydantic import BaseModel
from datetime import date
from typing import List, Optional

class Record(BaseModel):
    date: date
    diagnosis_codes: str
    procedure_codes: str
    submitted_cost: float
    paid_cost: float
    benefit_type: str

class PatientSummaryResponse(BaseModel):
    patient_id: str
    medical_summary: str
    total_submitted: float
    total_paid: float
    record_count: int
    records: List[Record]