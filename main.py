from fastapi import FastAPI
from modules.patient.routes import router as patient_router
from modules.operations.routes import router as operations_router
from modules.billing.routes import router as billing_router
from modules.insurance.routes import router as insurance_router
from huggingface_hub import login
from config.settings import get_settings

app = FastAPI(title="EHR Intelligence Framework", version="1.0")

settings = get_settings()
login(token=settings.HUGGING_FACE_API_KEY)

app.include_router(patient_router)
app.include_router(operations_router)
app.include_router(billing_router)
app.include_router(insurance_router)
@app.get("/")
async def root():
    return {"message": "EHR Framework API - Patient Summary Ready"}