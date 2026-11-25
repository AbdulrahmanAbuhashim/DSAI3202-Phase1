import json
from pathlib import Path

import score_catboost as original

defaults = None
VERSION = "road-accident-severity:3"


def init():
    global defaults
    original.init()
    path = Path(__file__).with_name("catboost_demo_defaults.json")
    defaults = json.loads(path.read_text())


def run(raw_data):
    try:
        request = (
            json.loads(raw_data)
            if isinstance(raw_data, (str, bytes))
            else raw_data
        )

        # Keep the original 21-feature API available.
        if "demo_input" not in request:
            return original.run(request)

        data = request["demo_input"]
        if not isinstance(data, dict):
            raise ValueError("demo_input must be a JSON object")

        required = [
            "speed_limit", "hour",
            "weather_conditions", "light_conditions"
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing demo fields: {missing}")

        speed = int(data["speed_limit"])
        hour = int(data["hour"])
        weather = str(data["weather_conditions"])
        lighting = str(data["light_conditions"])

        if speed not in (20, 30, 40, 50, 60, 70):
            raise ValueError("Unsupported UK speed limit")
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23")
        if weather not in ("1", "2"):
            raise ValueError("Weather must be 1 or 2")
        if lighting not in ("1", "4"):
            raise ValueError("Lighting must be 1 or 4")

        record = defaults.copy()
        record.update({
            "speed_limit": speed,
            "hour": hour,
            "weather_conditions": weather,
            "light_conditions": lighting,
            "is_night": int(hour < 6 or hour >= 20),
            "high_speed_road": int(speed >= 50)
        })

        result = original.run({"input_data": [record]})

        if "error" in result:
            return result

        result["severity"] = {
            1: "Fatal", 2: "Serious", 3: "Slight"
        }[result["predictions"][0]]
        result["mode"] = "simplified_demo"
        result["derived_features"] = {
            "is_night": record["is_night"],
            "high_speed_road": record["high_speed_road"]
        }
        result["assumed_features"] = defaults
        return result

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return {"error": str(e), "model_version": VERSION}
