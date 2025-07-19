from datetime import datetime
from zoneinfo import ZoneInfo  # Built-in since Python 3.9


def timestamp():
    # Get current time in UTC (timezone-aware)
    now_utc = datetime.now(tz=ZoneInfo("UTC"))

    # Convert to US Pacific Time (Los Angeles)
    now_pacific = now_utc.astimezone(ZoneInfo("America/Los_Angeles"))

    # Format timestamp
    timestamp = now_pacific.strftime("%Y-%m-%d_%H-%M-%S")

    return timestamp
