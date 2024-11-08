from fastapi import APIRouter, status
from ..schemas import Plate

router = APIRouter(prefix="/plate", tags=["Plates"])


@router.post("/", status_code=status.HTTP_201_CREATED)
def add_plate(plate: Plate):
    new_plate = plate.model_dump()
    return new_plate
