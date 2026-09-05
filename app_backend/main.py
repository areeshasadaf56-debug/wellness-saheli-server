"""
main.py

FastAPI backend for the Wellness Saheli Flutter app.

Exposes:
  /predict                      -- PCOS ML prediction (stateless, public)
  /signup, /signin, /logout,
  /reset_password                -- accounts + session tokens
  /profile/{user_id}             -- health profile, OWNER-ONLY (auth required)
  /chat                          -- AI check-in, auth required (protects the
                                     paid Groq API key from anonymous use)
  /conditions, /methods_reference,
  /effectiveness, /eligibility   -- contraceptive eligibility tool (stateless, public)

Run from the project root:
    uvicorn app_backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import auth
import chat
import database
import eligibility_data

APP_DEPLOYMENT_DIR = os.path.join("app_deployment")
MAX_PROFILE_BYTES = 300_000  # ~300 KB -- generous for the diary/profile JSON blob

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

# CORS: origins are wide open ("*") deliberately, but allow_credentials
# is OFF. Auth here is a Bearer token in the Authorization header, not
# a cookie -- so there's no session-cookie/CSRF risk that
# allow_credentials would normally guard against, and the previous
# combination (allow_origins="*" + allow_credentials=True) was both
# invalid per the CORS spec and an unnecessary risk surface.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creates all tables on first run; a no-op after that.
database.init_db()


def _client_key(request: Request) -> str:
    """Best-effort client identifier for rate limiting. Falls back to
    a constant if the client host isn't available (e.g. some test
    clients) -- rate limiting still applies, just shared across those
    callers rather than being a hole."""
    return request.client.host if request.client else "unknown"


def get_current_user_id(
    authorization: str | None = Header(default=None),
) -> int:
    """FastAPI dependency: verifies the `Authorization: Bearer <token>`
    header and returns the signed-in user's id. Raises 401 if missing,
    malformed, or the token is invalid/expired."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Please sign in again.",
        )
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return auth.verify_token(token)
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


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
    normalized = value.strip().capitalize()
    if normalized not in BINARY_ENCODING:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be 'Yes' or 'No', got '{value}'"
        )
    return BINARY_ENCODING[normalized]


def encode_cycle(value: str) -> int:
    normalized = value.strip().capitalize()
    if normalized not in CYCLE_ENCODING:
        raise HTTPException(
            status_code=422,
            detail=f"cycle_regularity must be 'Regular' or 'Irregular', got '{value}'"
        )
    return CYCLE_ENCODING[normalized]


def build_feature_vector(data: PCOSInput) -> np.ndarray:
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
    try:
        feature_vector = build_feature_vector(data)
        scaled_vector = scaler.transform(feature_vector)
        pred_class = model.predict(scaled_vector)[0]
        pred_proba = model.predict_proba(scaled_vector)[0][1]  # probability of class 1 (PCOS)
    except HTTPException:
        raise
    except Exception:
        # Never leak model/library internals in the response.
        raise HTTPException(
            status_code=500,
            detail="Could not process the prediction. Please check your inputs and try again.",
        )

    return PCOSOutput(
        prediction="PCOS Detected" if pred_class == 1 else "No PCOS Detected",
        pcos_probability=round(float(pred_proba), 4),
        model_used=metadata["model_name"],
    )


# =================================================================
# ACCOUNTS -- /signup, /signin, /logout, /reset_password
# =================================================================

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


class ResetPasswordRequest(BaseModel):
    email: str
    new_password: str


@app.post("/signup")
def signup(data: SignUpRequest, request: Request):
    try:
        result = auth.sign_up(data.name, data.email, data.password, _client_key(request))
        return result
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/signin")
def signin(data: SignInRequest, request: Request):
    try:
        result = auth.sign_in(data.email, data.password, _client_key(request))
        return result
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    """Best-effort: invalidates the session if a token was sent. Always
    returns ok so the app can clear its local state regardless."""
    if authorization and authorization.startswith("Bearer "):
        auth.invalidate_session(authorization.removeprefix("Bearer ").strip())
    return {"status": "ok"}


@app.post("/reset_password")
def reset_password_endpoint(data: ResetPasswordRequest, request: Request):
    try:
        auth.reset_password(data.email, data.new_password, _client_key(request))
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    # Always the same generic response, whether or not the email
    # existed -- see auth.reset_password's docstring.
    return {"status": "ok"}


# =================================================================
# HEALTH PROFILE -- /profile/{user_id}  (OWNER-ONLY)
# =================================================================

@app.get("/profile/{user_id}")
def get_profile(user_id: str, current_user_id: int = Depends(get_current_user_id)):
    if str(current_user_id) != user_id:
        raise HTTPException(
            status_code=403, detail="You don't have permission to access this profile."
        )
    profile = database.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile found for this user.")
    return profile


@app.put("/profile/{user_id}")
def put_profile(
    user_id: str,
    profile: dict,
    current_user_id: int = Depends(get_current_user_id),
):
    if str(current_user_id) != user_id:
        raise HTTPException(
            status_code=403, detail="You don't have permission to modify this profile."
        )

    profile["user_id"] = user_id
    if len(json.dumps(profile)) > MAX_PROFILE_BYTES:
        raise HTTPException(status_code=413, detail="Profile payload is too large.")

    database.upsert_profile(user_id, profile)
    return {"status": "ok"}


# =================================================================
# AI CHECK-IN -- /chat  (auth required)
# =================================================================

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    profile_context: dict | None = None


@app.post("/chat")
def chat_endpoint(
    data: ChatRequest,
    current_user_id: int = Depends(get_current_user_id),
):
    return chat.get_reply(data.message, data.history, data.profile_context)


# =================================================================
# CONTRACEPTIVE ELIGIBILITY -- /conditions, /methods_reference,
# /effectiveness, /eligibility (stateless, no personal data -- public)
# =================================================================

class EligibilityRequest(BaseModel):
    condition_ids: list[str]


@app.get("/conditions")
def get_conditions():
    return eligibility_data.list_conditions()


@app.get("/methods_reference")
def get_methods_reference():
    return eligibility_data.list_methods()


@app.get("/effectiveness")
def get_effectiveness():
    return eligibility_data.list_effectiveness()


@app.post("/eligibility")
def post_eligibility(data: EligibilityRequest):
    try:
        return eligibility_data.check_eligibility(data.condition_ids)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))