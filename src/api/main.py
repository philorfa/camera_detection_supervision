from fastapi import FastAPI
from .routers import people_router, plate_router, modifications_router

app = FastAPI()

app.include_router(people_router)
app.include_router(plate_router)
app.include_router(modifications_router)
