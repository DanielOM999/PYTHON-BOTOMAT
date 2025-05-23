from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Validate database URL
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Database setup
try:
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
except Exception as e:
    logger.error(f"Database connection failed: {str(e)}")
    raise

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

# Improved fine logic
def calculate_fine(speed: float, zone: float) -> str:
    overspeed = speed - zone
    if overspeed <= 0:
        return "No fine"
    elif overspeed <= 5:
        return "800 NOK"
    elif overspeed <= 10:
        return "2100 NOK"
    elif overspeed <= 15:
        return "3800 NOK"
    elif overspeed <= 20:
        return "5500 NOK"
    elif overspeed <= 25:
        return "7800 NOK"
    else:
        return "Criminal charges - police report required"

# POST: Save to DB
@app.post("/check_speed", status_code=status.HTTP_201_CREATED)
async def check_speed(data: SpeedData):
    db = SessionLocal()
    try:
        # Calculate fine and overspeed
        overspeed = data.speed - data.zone
        fine = calculate_fine(data.speed, data.zone)
        
        # Create report object
        report = Report(
            plate=data.plate,
            speed=data.speed,
            zone=data.zone,
            overspeed=overspeed,
            fine=fine,
        )
        
        # Database operations
        db.add(report)
        db.commit()
        db.refresh(report)
        
        logger.info(f"New report created: {report.id}")
        
        return {
            "id": report.id,
            "plate": report.plate,
            "speed": report.speed,
            "zone": report.zone,
            "overspeed": report.overspeed,
            "fine": report.fine,
            "timestamp": report.timestamp.isoformat()
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving to database"
        )
    finally:
        db.close()

# GET: From DB
@app.get("/reports")
async def get_reports(limit: int = 100):
    db = SessionLocal()
    try:
        results = db.query(Report).order_by(Report.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "plate": r.plate,
                "speed": r.speed,
                "zone": r.zone,
                "overspeed": r.overspeed,
                "fine": r.fine,
                "timestamp": r.timestamp.isoformat()
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving reports"
        )
    finally:
        db.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected" if engine else "disconnected"}
