from fastapi import APIRouter, status
from ..schemas import Vehicle

router = APIRouter(prefix="/modifications", tags=["Modifications"])


@router.post("/", status_code=status.HTTP_201_CREATED)
def add_vehicle(vehicle: Vehicle):
    new_vehicle = vehicle.model_dump()
    return new_vehicle