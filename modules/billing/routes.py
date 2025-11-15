# modules/billing/routes.py
from fastapi import APIRouter, Query
from modules.billing.service import run_billing_analysis, save_billing_results_to_supabase
from modules.billing.schemas import BillingValidationResponse, ClaimRecord

router = APIRouter(prefix="/billing", tags=["Billing"])

@router.get("/validate", response_model=BillingValidationResponse)
async def get_billing_validation(save: bool = Query(False)):
    result = run_billing_analysis()

    if save:
        save_billing_results_to_supabase(result)

    return BillingValidationResponse(
        invalid_claims=[ClaimRecord(**r) for r in result["invalid_claims"]],
        duplicate_claims=[ClaimRecord(**r) for r in result["duplicate_claims"]],
        long_turnaround_claims=[ClaimRecord(**r) for r in result["long_turnaround_claims"]],
        total_invalid=result["total_invalid"],
        total_duplicates=result["total_duplicates"],
        total_long_turnaround=result["total_long_turnaround"],
    )