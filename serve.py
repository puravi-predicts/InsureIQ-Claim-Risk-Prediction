import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Extra

app = FastAPI(
    title="InsureIQ: Predictive Claim Risk Scoring API",
    description="Production endpoint for evaluating rare-event insurance claim probabilities (14-phase pipeline framework).",
    version="1.0.0"
)

class ClaimFeaturePayload(BaseModel):
    ps_ind_01: int
    ps_car_13: float
    ps_reg_03: float
    
    class Config:
        extra = Extra.allow

model = None

@app.on_event("startup")
def load_production_artifacts():
    global model
    model_path = "model_artifacts/best_model_calibrated.joblib"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Successfully loaded InsureIQ calibrated production stack from {model_path}")
        except Exception as e:
            print(f"Error deserializing model binary: {str(e)}")
    else:
        print(f"Warning: Model artifact not found at '{model_path}'. Running in operational mock mode.")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_risk(payload: ClaimFeaturePayload):
    raw_inputs = payload.dict()
    
    if model is None:
        mock_probability = float(np.random.uniform(0.01, 0.08))
        return {
            "prediction_engine": "InsureIQ-Mock-Fallback",
            "claim_probability": round(mock_probability, 4),
            "decision_threshold": "Cost-Optimal (Dynamic)",
            "risk_category": "High Risk" if mock_probability > 0.036 else "Standard Risk"
        }
    
    try:
        input_df = pd.DataFrame([raw_inputs])
        probability = float(model.predict_proba(input_df)[0][1])
        BUSINESS_THRESHOLD = 0.0364
        
        return {
            "prediction_engine": "InsureIQ-Calibrated-Stack-V1",
            "claim_probability": round(probability, 4),
            "business_threshold_applied": BUSINESS_THRESHOLD,
            "risk_category": "High Risk (Flagged for Adjuster Review)" if probability >= BUSINESS_THRESHOLD else "Standard Risk (Auto-Pass)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(e)}")