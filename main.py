from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SpeedData(BaseModel):
    plate: str
    speed: float
    zone: int

# Store all reports in memory
reports = []

# Fine calculation logic
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

# POST: New report
@app.post("/check_speed")
def check_speed(data: SpeedData):
    overspeed = data.speed - data.zone
    fine = calculate_fine(data.speed, data.zone)
    result = {
        "plate": data.plate,
        "speed": data.speed,
        "zone": data.zone,
        "overspeed": overspeed,
        "fine": fine,
        "timestamp": datetime.now().isoformat()
    }
    reports.append(result)
    return result

# GET: All reports
@app.get("/reports")
def get_reports():
    return reports[::-1]  # Most recent first
