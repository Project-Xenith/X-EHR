# modules/operations/routes.py
from fastapi import APIRouter, Query
from datetime import date
from typing import Optional
from modules.operations.service import get_operational_insights, save_insights_to_supabase
from modules.operations.schemas import OperationalInsightsResponse, TopItem

router = APIRouter(prefix="/operations", tags=["Operations"])

@router.get("/insights", response_model=OperationalInsightsResponse)
async def get_insights(
    start_date: date = Query(..., description="Quarter start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="Quarter end date (YYYY-MM-DD)"),
    top_n: Optional[int] = Query(5, ge=1, le=20),
    save: bool = Query(False, description="Save to Supabase?")
):
    _, insights = get_operational_insights(start_date, end_date, top_n)

    if save:
        save_insights_to_supabase(insights)

    return OperationalInsightsResponse(
        quarter=insights["quarter"],
        total_active_patients=insights["total_active_patients"],
        top_diagnoses=[
            TopItem(code=code, count=count)
            for code, count in insights["top_diagnoses"].items()
        ],
        top_procedures=[
            TopItem(code=code, count=count)
            for code, count in insights["top_procedures"].items()
        ]
    )