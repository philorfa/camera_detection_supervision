from datetime import datetime
import pytz
from src.core.settings import get_settings

settings = get_settings()


def get_date_time_dict(utc: bool = True,
                       location: str = settings.DRONE_LOCATION) -> dict:
    """Returns a dictionary with the current date and time formatted as \
        strings."""

    utc_time = get_utc_time() if utc else get_local_time(location)

    current_day = utc_time.strftime("%d-%m-%Y")
    current_time = utc_time.strftime("%H:%M:%S")

    return {"day": current_day, "time": current_time}


def get_utc_time() -> datetime:
    """Returns the current UTC time as a datetime object without \
        microseconds."""
    return datetime.utcnow().replace(microsecond=0)


def get_local_time(location: str = settings.DRONE_LOCATION) -> datetime:
    """Returns the current local time for the specified location."""
    specified_timezone = pytz.timezone(location)
    return datetime.now(specified_timezone).replace(tzinfo=None, microsecond=0)
