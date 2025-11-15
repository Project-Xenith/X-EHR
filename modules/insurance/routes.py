# modules/insurance/routes.py
from fastapi import APIRouter, Query
from modules.insurance.service import run_insurance_analysis, simulate_plan, load_and_prepare, engineer_features, aggregate_member_level
from modules.insurance.schemas import InsuranceModelResponse, SimulatePlanRequest, PlanSimulation

router = APIRouter(prefix="/insurance", tags=["Insurance"])

@router.get("/model", response_model=InsuranceModelResponse)
async def get_insurance_model(retrain: bool = Query(False)):
    if retrain:
        # Force retrain
        import os
        if os.path.exists("ai_models/insurance_glm.pkl"):
            os.remove("ai_models/insurance_glm.pkl")
    return run_insurance_analysis()

@router.post("/simulate", response_model=PlanSimulation)
async def simulate_custom_plan(req: SimulatePlanRequest):
    df = load_and_prepare()
    df = engineer_features(df)
    member_df = aggregate_member_level(df)
    return simulate_plan(member_df, req.deductible, req.coinsurance, req.oop_max, req.plan_name)