from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_data = None

REQUIRED_COLUMNS = [
    "Student_ID", "First_Name", "Last_Name", "Email", "Gender", "Age",
    "Department", "Attendance (%)", "Midterm_Score", "Final_Score",
    "Assignments_Avg", "Quizzes_Avg", "Participation_Score", "Projects_Score",
    "Total_Score", "Grade", "Study_Hours_per_Week", "Extracurricular_Activities",
    "Internet_Access_at_Home", "Parent_Education_Level", "Family_Income_Level",
    "Stress_Level (1-10)", "Sleep_Hours_per_Night", "Total_Score_Recalculated"
]

@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    global uploaded_data
    contents = await file.read()
    filename = file.filename.lower()

    try:
        # Read file
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .csv or .xlsx file.")

        # Normalize column headers
        df.columns = [col.strip() for col in df.columns]
        print("Uploaded file columns:", df.columns.tolist())

        # Strict match check
        if list(df.columns) != REQUIRED_COLUMNS:
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]

            print("Missing columns:", missing)
            print("Extra columns:", extra)

            error_lines = ["❌ Upload Failed:\n"]

            if missing:
                error_lines.append("Missing columns:")
                error_lines += [f"- {col}" for col in missing]

            if extra:
                error_lines.append("\nUnexpected columns:")
                error_lines += [f"- {col}" for col in extra]

            error_lines.append("\nPlease ensure your file includes all required columns and no extra ones.")

            error_msg = "\n".join(error_lines)
            raise HTTPException(status_code=400, detail=error_msg)

        # Store uploaded data
        uploaded_data = df

        return {
            "message": "File uploaded successfully.",
            "rows": len(df),
            "columns": len(df.columns)
        }

    except Exception as e:
        print("Upload error:", str(e))
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")


@app.get("/dashboard/summary")
def get_summary():
    global uploaded_data
    if uploaded_data is None:
        return {"error": "No data uploaded yet."}

    try:
        avg_score = uploaded_data['Final_Score'].mean()
        at_risk_count = len(uploaded_data[uploaded_data['Final_Score'] < 50])
        return {
            "average_score": round(avg_score, 2),
            "at_risk_students": at_risk_count,
        }
    except Exception as e:
        print("Summary error:", str(e))
        return {"error": f"Failed to generate summary: {str(e)}"}


@app.get("/dashboard/charts-data")
def get_charts_data():
    global uploaded_data
    if uploaded_data is None:
        return []

    # Replace NaN with None for JSON compatibility
    safe_data = uploaded_data.where(pd.notnull(uploaded_data), None)
    return safe_data.to_dict(orient="records")

@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    global uploaded_data
    contents = await file.read()
    filename = file.filename.lower()

    try:
        # Read file
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .csv or .xlsx file.")

        # Normalize columns and validation here as before...

        # Rename columns for model input if needed
        if 'Attendance' in df.columns:
            df['Attendance (%)'] = df.pop('Attendance')
        if 'Stress_Level' in df.columns:
            df['Stress_Level (1-10)'] = df.pop('Stress_Level')

        # Store uploaded data
        uploaded_data = df

        # Perform bulk prediction if model loaded
        if model is not None:
            try:
                predictions = model.predict(df)
                df['Predicted_Score'] = predictions
                df['Risk_Level'] = df['Predicted_Score'].apply(lambda x: 'At Risk' if x < 50 else 'Not At Risk')
                uploaded_data = df  # Update global with prediction results included
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        return {
            "message": "File uploaded and predictions generated successfully.",
            "rows": len(df),
            "columns": len(df.columns),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")


