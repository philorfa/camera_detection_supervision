from pydantic import BaseModel
from datetime import datetime
from typing import List


class Vehicle(BaseModel):
    original_frames: List[str]
    cropped_frames: List[str]
    car_color: str
    color_confidence: float
    color_reasoning: str
    car_type: str
    type_confidence: float
    type_reasoning: str
    detected_at: datetime
