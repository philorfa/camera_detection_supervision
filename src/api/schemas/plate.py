from pydantic import BaseModel
from datetime import datetime
from typing import List


class Plate(BaseModel):
    frame: str
    xai_frame: str
    car_coordinates: List[float]
    plate_coordinates: List[float]
    plate: str
    plate_confidence: float
    detected_at: datetime
