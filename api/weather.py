import requests_cache
from datetime import timedelta

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
LAT=25.873200918122755
LONG=82.09919645244908

def get_weather_data(latitude=LAT, longitude=LONG):
    session = requests_cache.CachedSession('.cache', expire_after=timedelta(hours=1))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation",
        "current_weather": True,
        "forecast_days": 2,
        "timezone": "auto",
    }
    response = session.get(OPEN_METEO_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    current = data.get("current_weather", {})
    hourly = data.get("hourly", {})
    hourly_time = hourly.get("time", [])
    precipitation = hourly.get("precipitation", [])
    humidity = hourly.get("relativehumidity_2m", [])
    temperatures = hourly.get("temperature_2m", [])

    current_time = current.get("time")
    current_index = 0
    if current_time and current_time in hourly_time:
        current_index = hourly_time.index(current_time)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": data.get("timezone", ""),
        "temperature": current.get("temperature"),
        "wind_speed": current.get("windspeed"),
        "wind_direction": current.get("winddirection"),
        "weather_code": current.get("weathercode"),
        "humidity": humidity[current_index] if current_index < len(humidity) else None,
        "precipitation": precipitation[current_index] if current_index < len(precipitation) else None,
        "temperature_hourly": temperatures,
        "time_hourly": hourly_time,
        "raw": data,
    }
