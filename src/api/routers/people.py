from fastapi import APIRouter, status
from ..schemas import People

router = APIRouter(prefix="/people", tags=["People"])


@router.post("/", status_code=status.HTTP_201_CREATED)
def add_people(people: People):
    new_people = people.model_dump()
    return new_people
