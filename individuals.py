from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import joblib

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = joblib.load("student_performance_pipeline.joblib")
    print("✅ Model loaded for student insights.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# Fixed module mapping per department
department_modules = {
    "Computer Science": ["DSA", "AI", "DBMS", "CN", "OS"],
    "Business": ["Accounting", "Finance", "Marketing", "HR", "Economics"],
    "Engineering": ["Mechanics", "Thermodynamics", "Materials", "Circuits", "Maths"],
    "Education": ["Curriculum", "Psychology", "Sociology", "Assessment", "Teaching Methods"]
}

# Global to hold uploaded student data
uploaded_data = pd.DataFrame()

@app.post("/student/insights")
async def student_insights(file: UploadFile = File(...)):
    global uploaded_data

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # Required columns
        required_cols = [
            "Student_ID", "First_Name", "Last_Name", "Department",
            "Age", "Gender", "Study_Hours_per_Week", "Stress_Level (1-10)",
            "Sleep_Hours_per_Night", "Participation_Score", "Projects_Score",
            "Attendance (%)", "Midterm_Score", "Final_Score", "Quizzes_Avg", "Assignments_Avg",
            "Grade", "Internet_Access_at_Home", "Extracurricular_Activities", 
            "Parent_Education_Level", "Family_Income_Level"
        ]

        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail="Missing required columns.")

        dept = df.iloc[0]['Department']
        modules = department_modules.get(dept)
        if not modules:
            raise HTTPException(status_code=400, detail=f"Unknown department '{dept}'.")

        # Predictions
        input_features = df.drop(columns=["Student_ID", "First_Name", "Last_Name"])
        predictions = model.predict(input_features)
        df['Predicted_Score'] = predictions

        # Comparison to class average
        class_avg = df['Final_Score'].mean()
        df['Compared_to_Class_Avg'] = df['Final_Score'].apply(
            lambda x: "Above Average" if x > class_avg else ("Below Average" if x < class_avg else "Average")
        )

        # Roadmap
        def roadmap(row):
            tips = []
            if row['Attendance (%)'] < 75:
                tips.append("Improve class attendance.")
            if row['Study_Hours_per_Week'] < 10:
                tips.append("Increase study hours.")
            if row['Predicted_Score'] < 50:
                tips.append("Seek academic support.")
            if not tips:
                tips.append("Maintain current effort and stay consistent.")
            return tips

        df['Improvement_Roadmap'] = df.apply(roadmap, axis=1)

        uploaded_data = df  # Save uploaded data for later retrieval

        insights = df[["Student_ID", "First_Name", "Last_Name", "Predicted_Score",
                       "Compared_to_Class_Avg", "Improvement_Roadmap"]].to_dict(orient="records")

        return {"department": dept, "modules": modules, "insights": insights}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/student-insights/{student_id}")
def get_student_insights(student_id: str):
    global uploaded_data

    if uploaded_data.empty:
        raise HTTPException(status_code=404, detail="No data uploaded yet.")

    if student_id not in uploaded_data["Student_ID"].astype(str).values:
        raise HTTPException(status_code=404, detail="Student not found.")

    student_row = uploaded_data[uploaded_data["Student_ID"].astype(str) == student_id].iloc[0]

    skill_columns = [
        "Midterm_Score", "Final_Score", "Assignments_Avg",
        "Quizzes_Avg", "Participation_Score", "Projects_Score"
    ]

    student_scores = {col: float(student_row[col]) for col in skill_columns}
    class_averages = uploaded_data[skill_columns].mean().round(2).to_dict()

    threshold = uploaded_data["Final_Score"].quantile(0.9)
    top_students = uploaded_data[uploaded_data["Final_Score"] >= threshold]
    top_averages = top_students[skill_columns].mean().round(2).to_dict()

    improvement_areas = {}
    for col in skill_columns:
        student_val = student_scores[col]
        top_val = top_averages[col]
        if top_val == 0:
            continue
        percentage = (student_val / top_val) * 100
        improvement_areas[col] = round(percentage, 1)

    return {
        "student_id": student_id,
        "student_scores": student_scores,
        "class_averages": class_averages,
        "top_performer_averages": top_averages,
        "improvement_percentages": improvement_areas
    }

