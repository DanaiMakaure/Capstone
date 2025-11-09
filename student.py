from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "student_data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_student_file_path(student_number: str) -> str:
    safe_student = student_number.replace("/", "_")  # basic sanitize
    return os.path.join(DATA_DIR, f"{safe_student}.json")

def load_student_data(student_number: str) -> pd.DataFrame:
    path = get_student_file_path(student_number)
    if os.path.exists(path):
        with open(path, "r") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    else:
        return pd.DataFrame()

def save_student_data(student_number: str, df: pd.DataFrame):
    path = get_student_file_path(student_number)
    df.to_json(path, orient="records")

def recalc_avg_feedback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['Module Name', 'Type', 'Number']).reset_index(drop=True)
    df['AVG'] = 0.0
    df['Feedback'] = ''

    grouped = df.groupby(['Module Name', 'Type'])
    for (module, typ), group in grouped:
        scores = []
        idxs = group.index.tolist()
        for i, idx in enumerate(idxs):
            score = df.at[idx, 'Score']
            scores.append(score)
            avg = sum(scores) / len(scores)
            df.at[idx, 'AVG'] = round(avg, 1)
            df.at[idx, 'Feedback'] = "At Risk" if avg < 55 else "Not at Risk"
    return df

@app.post("/upload_student_marks/")
async def upload_student_marks(
    student_number: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            new_df = pd.read_csv(BytesIO(content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            new_df = pd.read_excel(BytesIO(content))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type. Upload a .csv or .xlsx file."})

        required_cols = {'Full Name', 'Module Name', 'Type', 'Number', 'Score'}
        if not required_cols.issubset(new_df.columns):
            return JSONResponse(status_code=400, content={"error": f"File must contain columns: {', '.join(required_cols)}"})

        new_df['Student Number'] = student_number

        existing_df = load_student_data(student_number)
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=['Student Number', 'Full Name', 'Module Name', 'Type', 'Number'],
                keep='last'
            )
        else:
            combined_df = new_df

        combined_df = recalc_avg_feedback(combined_df)
        combined_df = combined_df[['Student Number', 'Full Name', 'Module Name', 'Type', 'Number', 'Score', 'AVG', 'Feedback']]
        save_student_data(student_number, combined_df)

        return {
            "data": combined_df.to_dict(orient="records"),
            "rows": len(combined_df),
            "columns": len(combined_df.columns),
            "message": f"Upload successful. {len(combined_df)} rows processed for student {student_number}."
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

@app.get("/get_student_data/{student_number}")
async def get_student_data(student_number: str):
    try:
        df = load_student_data(student_number)
        return {
            "data": df.to_dict(orient="records"),
            "message": f"{'Loaded data' if not df.empty else 'No data found'} for student {student_number}."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

