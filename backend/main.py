# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# === IMPORT ALL ROUTERS ===
from modules.patient.routes import router as patient_router
from modules.operations.routes import router as operations_router
from modules.billing.routes import router as billing_router
from modules.insurance.routes import router as insurance_router

# === OPTIONAL: old single-file modules (if you still want them) ===
# from billing_department import router as billing_old_router
# from insurance_design import router as insurance_old_router

app = FastAPI(
    title="EHR Intelligence Platform",
    version="1.0.0",
    description="Patient • Operations • Billing • Insurance"
)

# === CORS: Allow frontend to call backend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === INCLUDE ROUTERS ===
app.include_router(patient_router)
app.include_router(operations_router)
app.include_router(billing_router)
app.include_router(insurance_router)

# Optional old routers
# app.include_router(billing_old_router)
# app.include_router(insurance_old_router)

@app.get("/")
async def root():
    return {"message": "EHR Platform API – All modules loaded"}

# === Run with uvicorn ===
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )