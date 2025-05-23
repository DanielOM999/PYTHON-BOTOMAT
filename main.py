from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    plate = Column(String, index=True)
    speed = Column(Float)
    zone = Column(Integer)
    overspeed = Column(Float)
    fine = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class SpeedData(BaseModel):
    plate: str
    speed: float
    zone: int

# Fine logic
def calculate_fine(speed, zone):
    over_speed = speed - zone
    if over_speed <= 0:
        return 0
    elif over_speed <= 5:
        return 800
    elif over_speed <= 10:
        return 2100
    elif over_speed <= 15:
        return 3800
    elif over_speed <= 20:
        return 5500
    elif over_speed <= 25:
        return 7800
    else:
        return "Too fast – police report"

# POST: Save to DB
@app.post("/check_speed")
def check_speed(data: SpeedData):
    db = SessionLocal()
    try:
        overspeed = data.speed - data.zone
        fine = calculate_fine(data.speed, data.zone)
        report = Report(
            plate=data.plate,
            speed=data.speed,
            zone=data.zone,
            overspeed=overspeed,
            fine=str(fine),  # Convert to string for consistency
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return {
            "plate": report.plate,
            "speed": report.speed,
            "zone": report.zone,
            "overspeed": report.overspeed,
            "fine": report.fine,
            "timestamp": report.timestamp.isoformat()
        }
    finally:
        db.close()

# GET: From DB
@app.get("/reports")
def get_reports():
    db = SessionLocal()
    try:
        return [
            {
                "plate": r.plate,
                "speed": r.speed,
                "zone": r.zone,
                "overspeed": r.overspeed,
                "fine": r.fine,
                "timestamp": r.timestamp.isoformat()
            }
            for r in db.query(Report).order_by(Report.timestamp.desc()).all()
        ]
    finally:
        db.close()
