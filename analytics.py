from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataPoint(BaseModel):
    x: float
    y: float
    date: str

attendance_data_store = []
performance_data_store = []

@app.post("/api/save_attendance")
async def save_attendance(data: List[DataPoint]):
    global attendance_data_store
    attendance_data_store = data
    return {"status": "success", "message": "Attendance saved"}

@app.post("/api/save_performance")
async def save_performance(data: List[DataPoint]):
    global performance_data_store
    performance_data_store = data
    return {"status": "success", "message": "Performance saved"}

@app.get("/api/get_attendance")
async def get_attendance():
    return attendance_data_store

@app.get("/api/get_performance")
async def get_performance():
    return performance_data_store

