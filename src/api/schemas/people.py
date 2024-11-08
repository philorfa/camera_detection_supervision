from pydantic import BaseModel
from datetime import datetime
from typing import List


class People(BaseModel):
    frame: str
    xai_frame:str
    car_coordinates: List[float]
    people: int
    people_coordinates: List[List[float]]
    people_confidence: List[float]
    detected_at: datetime
