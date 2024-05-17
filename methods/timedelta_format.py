import datetime
import math


def timedelta_format(delta: datetime.timedelta) -> str:
    hours = math.trunc(delta.total_seconds() / 3600)
    seconds_minus_hours = delta.total_seconds() - hours * 3600
    minutes = math.trunc(seconds_minus_hours / 60)
    seconds_minus_minutes = seconds_minus_hours - minutes * 60
    return f"{hours:02}:{minutes:02}:{seconds_minus_minutes:08.5f}"

