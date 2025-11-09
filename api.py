from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Allow all origins for testing (change to specific frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model with error handling
try:
    model = joblib.load("student_performance_pipeline.joblib")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

class InputData(BaseModel):
    Age: int
    Quizzes_Avg: float
    Final_Score: float
    Study_Hours_per_Week: float
    Stress_Level: int  # scale 1-10
    Projects_Score: float
    Participation_Score: float
    Sleep_Hours_per_Night: float
    Attendance: float  # will rename to 'Attendance (%)'
    Midterm_Score: float
    Assignments_Avg: float

    Gender: str
    Department: str
    Extracurricular_Activities: str
    Internet_Access_at_Home: str
    Parent_Education_Level: str
    Family_Income_Level: str
    Grade: str

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not loaded."}

    input_dict = data.dict()

    # Rename keys to match model expected columns
    input_dict['Attendance (%)'] = input_dict.pop('Attendance')
    input_dict['Stress_Level (1-10)'] = input_dict.pop('Stress_Level')

    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

    advice = []
    if input_dict['Attendance (%)'] < 75:
        advice.append("Your attendance is below average. Try to attend more classes.")
    if input_dict['Study_Hours_per_Week'] < 10:
        advice.append("Consider increasing your study hours to improve your score.")
    if prediction < 50:
        advice.append("You are currently at risk. Seek additional help and support.")
    else:
        advice.append("Keep up the good work!")

    return {
        "predicted_score": round(prediction, 2),
        "feedback": advice
    }

