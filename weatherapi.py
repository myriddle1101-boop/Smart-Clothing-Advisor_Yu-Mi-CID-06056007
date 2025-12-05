from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")

print("You weather API Key is：", API_KEY)

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")


def get_london_weather():
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q=London&appid={API_KEY}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print("Request unsuccessfully：", data)
        return

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    weather = data["weather"][0]["description"]

    print("London weather：")
    print("temperature：", temp, "°C")
    print("feel_like weather：", feels_like, "°C")
    print("Weather condition：", weather)

    return {
        "temp": temp,
        "feels_like": feels_like,
        "weather": weather
    }

if __name__ == "__main__":
    get_london_weather()
