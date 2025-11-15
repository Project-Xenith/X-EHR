# modules/billing/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class ClaimRecord(BaseModel):
    claim_id: Optional[str] = None
    total_submitted_cost: Optional[float] = None
    total_paid_cost: Optional[float] = None
    total_patient_cost: Optional[float] = None
    proc_procedure_code: Optional[str] = None
    med_date_service: Optional[date] = None
    med_date_service_end: Optional[date] = None
    proc_date_service: Optional[date] = None
    turnaround_days: Optional[int] = None

class BillingValidationResponse(BaseModel):
    invalid_claims: List[ClaimRecord]
    duplicate_claims: List[ClaimRecord]
    long_turnaround_claims: List[ClaimRecord]
    total_invalid: int
    total_duplicates: int
    total_long_turnaround: int