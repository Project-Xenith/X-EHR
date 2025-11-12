# modules/operations/schemas.py
from pydantic import BaseModel
from typing import Dict, List
from datetime import date

class TopItem(BaseModel):
    code: str
    count: int

class OperationalInsightsResponse(BaseModel):
    quarter: str
    total_active_patients: int
    top_diagnoses: List[TopItem]
    top_procedures: List[TopItem]

class OperationalInsightsDB(BaseModel):
    quarter: str
    top_diagnoses: Dict[str, int]
    top_procedures: Dict[str, int]
    total_active_patients: int