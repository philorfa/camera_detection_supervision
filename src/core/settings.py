from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os


class Settings(BaseSettings):
    APP_NAME: str = ""
    ENV: str = ""
    VERSION: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    ROBOFLOW_API_KEY: str = ""
    BASE_URL: str = ""
    PEOPLE_ENDPOINT: str = ""
    PLATE_ENDPOINT: str = ""
    MODIFICATIONS_ENDPOINT: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = ""
    VEHICLES: List = []
    PEOPLE: List = []
    SCALE: int = 2
    DRONE_LOCATION: str = ""

    @property
    def people_url(self) -> str:
        return f"{self.BASE_URL}{self.PEOPLE_ENDPOINT}"

    @property
    def plate_url(self) -> str:
        return f"{self.BASE_URL}{self.PLATE_ENDPOINT}"
    
    @property
    def properties_url(self) -> str:
        return f"{self.BASE_URL}{self.MODIFICATIONS_ENDPOINT}"

    model_config = ConfigDict(env_file=f".env.{os.getenv('ENV', 'dev')}",
                              extra="allow")


@lru_cache()
def get_settings():
    return Settings()
