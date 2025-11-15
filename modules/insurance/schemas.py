# modules/insurance/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import date

class RatingFactor(BaseModel):
    feature: str
    coefficient: float
    factor: float

class AgeBand(BaseModel):
    age: int
    predicted_pmpm: float
    relative_to_base: float

class PlanSimulation(BaseModel):
    plan: str
    deductible: float
    coinsurance: float
    oop_max: float
    avg_oop_per_member: float
    avg_plan_cost_pmpm: float
    total_plan_cost: float

class InsuranceModelResponse(BaseModel):
    base_rate: float
    mae: float
    r2: float
    top_rating_factors: List[RatingFactor]
    age_rating_table: List[AgeBand]
    sample_plans: List[PlanSimulation]

class SimulatePlanRequest(BaseModel):
    deductible: float = 0
    coinsurance: float = 0.2
    oop_max: float = 5000
    plan_name: Optional[str] = "Custom"