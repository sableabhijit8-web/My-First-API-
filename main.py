
from pathlib import Path
from typing import Annotated, Literal
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field


# -----------------------------
# Load the trained ML model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


app = FastAPI(
    title="Insurance Premium Prediction API",
    description="Predicts an insurance premium category using user details.",
    version="1.0.0"
)


# -----------------------------
# City classification
# -----------------------------
tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Hyderabad", "Pune"
]

tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna",
    "Ranchi", "Visakhapatnam", "Coimbatore", "Bhopal",
    "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur",
    "Raipur", "Amritsar", "Varanasi", "Agra", "Dehradun",
    "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram",
    "Ludhiana", "Nashik", "Allahabad", "Udaipur",
    "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada",
    "Tiruchirappalli", "Bhavnagar", "Gwalior", "Dhanbad",
    "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur",
    "Asansol", "Siliguri"
]


# -----------------------------
# Input validation
# -----------------------------
class UserInput(BaseModel):
    age: Annotated[
        int, Field(gt=0, lt=120, description="Age in years")
    ]

    weight: Annotated[
        float, Field(gt=0, description="Weight in kilograms")
    ]

    height: Annotated[
        float, Field(gt=0, lt=2.5, description="Height in meters")
    ]

    income_lpa: Annotated[
        float, Field(gt=0, description="Annual income in lakhs")
    ]

    smoker: bool

    city: Annotated[
        str, Field(min_length=1, description="City of residence")
    ]

    occupation: Literal[
        "retired",
        "freelancer",
        "student",
        "government_job",
        "business_owner",
        "unemployed",
        "private_job"
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        city = self.city.strip().casefold()

        if city in [c.casefold() for c in tier_1_cities]:
            return 1
        elif city in [c.casefold() for c in tier_2_cities]:
            return 2
        return 3


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Insurance Premium Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_premium(data: UserInput):
    input_df = pd.DataFrame([{
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation
    }])

    # Predict category
    prediction = model.predict(input_df)[0]

    # Calculate class probabilities when supported
    confidence = None
    class_probabilities = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        classes = getattr(model, "classes_", range(len(probabilities)))

        class_probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities)
        }

        confidence = round(float(max(probabilities)), 4)

    return {
        "response": {
            "predicted_category": str(prediction),
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "user_details": {
                "bmi": data.bmi,
                "age_group": data.age_group,
                "lifestyle_risk": data.lifestyle_risk,
                "city_tier": data.city_tier
            }
        }
    }
