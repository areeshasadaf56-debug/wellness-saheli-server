"""
main.py

FastAPI backend that serves the app-matched PCOS model
(app_deployment/pcos_app_model.joblib) to the Flutter app.

Exposes a single POST endpoint /predict that accepts the 22 form
fields (raw, human-readable values), applies the same scaling used
during training, runs the model, and returns a prediction + probability.

Run from the project root:
    uvicorn app_backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DEPLOYMENT_DIR = os.path.join("app_deployment")

# ---- Load model, scaler, and metadata once at startup ----
model = joblib.load(os.path.join(APP_DEPLOYMENT_DIR, "pcos_app_model.joblib"))
scaler = joblib.load(os.path.join(APP_DEPLOYMENT_DIR, "pcos_app_scaler.joblib"))

with open(os.path.join(APP_DEPLOYMENT_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]
CYCLE_ENCODING = metadata["categorical_encodings"]["Cycle(R/I)"]
BINARY_FIELDS = metadata["categorical_encodings"]["binary_yes_no_fields"]
BINARY_ENCODING = metadata["categorical_encodings"]["binary_encoding"]

app = FastAPI(title="PCOS Detection API", version="1.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PCOSInput(BaseModel):
    """
    Request body the Flutter app must send. All fields use plain,
    human-readable values -- the API handles converting Yes/No and
    Regular/Irregular into the numbers the model expects.
    """
    age_yrs: float = Field(..., description="Age in years")
    weight_kg: float = Field(..., description="Weight in kilograms")
    height_cm: float = Field(..., description="Height in centimeters")
    cycle_regularity: str = Field(..., description="'Regular' or 'Irregular'")
    cycle_length_days: float = Field(..., description="Average cycle length in days")
    prl: float = Field(..., description="PRL (ng/mL)")
    vit_d3: float = Field(..., description="Vitamin D3 (ng/mL)")
    prg: float = Field(..., description="PRG (ng/mL)")
    rbs: float = Field(..., description="Random Blood Sugar (mg/dl)")
    bp_systolic: float = Field(..., description="BP Systolic (mmHg)")
    bp_diastolic: float = Field(..., description="BP Diastolic (mmHg)")
    follicle_no_l: float = Field(..., description="Follicle No. (Left)")
    follicle_no_r: float = Field(..., description="Follicle No. (Right)")
    avg_f_size_l: float = Field(..., description="Avg. Follicle size (Left) mm")
    avg_f_size_r: float = Field(..., description="Avg. Follicle size (Right) mm")
    endometrium: float = Field(..., description="Endometrium thickness (mm)")
    weight_gain: str = Field(..., description="'Yes' or 'No'")
    hair_growth: str = Field(..., description="'Yes' or 'No'")
    skin_darkening: str = Field(..., description="'Yes' or 'No'")
    hair_loss: str = Field(..., description="'Yes' or 'No'")
    pimples: str = Field(..., description="'Yes' or 'No'")
    fast_food: str = Field(..., description="'Yes' or 'No'")
    regular_exercise: str = Field(..., description="'Yes' or 'No'")


class PCOSOutput(BaseModel):
    """Response returned to the Flutter app."""
    prediction: str
    pcos_probability: float
    model_used: str


def encode_binary(value: str, field_name: str) -> int:
    """
    Converts a 'Yes'/'No' string into the 1/0 the model expects.

    Args:
        value (str): 'Yes' or 'No' (case-insensitive).
        field_name (str): name of the field, used only for error messages.

    Returns:
        int: 1 for Yes, 0 for No.
    """
    normalized = value.strip().capitalize()
    if normalized not in BINARY_ENCODING:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be 'Yes' or 'No', got '{value}'"
        )
    return BINARY_ENCODING[normalized]


def encode_cycle(value: str) -> int:
    """
    Converts 'Regular'/'Irregular' into the 2/4 the model expects.

    Args:
        value (str): 'Regular' or 'Irregular' (case-insensitive).

    Returns:
        int: 2 for Regular, 4 for Irregular.
    """
    normalized = value.strip().capitalize()
    if normalized not in CYCLE_ENCODING:
        raise HTTPException(
            status_code=422,
            detail=f"cycle_regularity must be 'Regular' or 'Irregular', got '{value}'"
        )
    return CYCLE_ENCODING[normalized]


def build_feature_vector(data: PCOSInput) -> np.ndarray:
    """
    Converts the incoming request into a feature vector in the EXACT
    order the model was trained on (FEATURE_ORDER from metadata).

    Args:
        data (PCOSInput): validated request body.

    Returns:
        numpy.ndarray: shape (1, 22), ready to be scaled and predicted on.
    """
    bmi = data.weight_kg / ((data.height_cm / 100) ** 2)

    values_by_name = {
        "Age (yrs)": data.age_yrs,
        "BMI": bmi,
        "Cycle(R/I)": encode_cycle(data.cycle_regularity),
        "Cycle length(days)": data.cycle_length_days,
        "PRL(ng/mL)": data.prl,
        "Vit D3 (ng/mL)": data.vit_d3,
        "PRG(ng/mL)": data.prg,
        "RBS(mg/dl)": data.rbs,
        "BP _Systolic (mmHg)": data.bp_systolic,
        "BP _Diastolic (mmHg)": data.bp_diastolic,
        "Follicle No. (L)": data.follicle_no_l,
        "Follicle No. (R)": data.follicle_no_r,
        "Avg. F size (L) (mm)": data.avg_f_size_l,
        "Avg. F size (R) (mm)": data.avg_f_size_r,
        "Endometrium (mm)": data.endometrium,
        "Weight gain(Y/N)": encode_binary(data.weight_gain, "weight_gain"),
        "hair growth(Y/N)": encode_binary(data.hair_growth, "hair_growth"),
        "Skin darkening (Y/N)": encode_binary(data.skin_darkening, "skin_darkening"),
        "Hair loss(Y/N)": encode_binary(data.hair_loss, "hair_loss"),
        "Pimples(Y/N)": encode_binary(data.pimples, "pimples"),
        "Fast food (Y/N)": encode_binary(data.fast_food, "fast_food"),
        "Reg.Exercise(Y/N)": encode_binary(data.regular_exercise, "regular_exercise"),
    }

    ordered_values = [values_by_name[feature_name] for feature_name in FEATURE_ORDER]
    return np.array(ordered_values, dtype=float).reshape(1, -1)


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "model": metadata["model_name"]}


@app.post("/predict", response_model=PCOSOutput)
def predict(data: PCOSInput):
    """
    Runs the PCOS model on the submitted form data.

    Args:
        data (PCOSInput): the 22 form fields from the Flutter app.

    Returns:
        PCOSOutput: prediction label, PCOS probability, and model name.
    """
    feature_vector = build_feature_vector(data)
    scaled_vector = scaler.transform(feature_vector)

    pred_class = model.predict(scaled_vector)[0]
    pred_proba = model.predict_proba(scaled_vector)[0][1]  # probability of class 1 (PCOS)

    return PCOSOutput(
        prediction="PCOS Detected" if pred_class == 1 else "No PCOS Detected",
        pcos_probability=round(float(pred_proba), 4),
        model_used=metadata["model_name"],
    )