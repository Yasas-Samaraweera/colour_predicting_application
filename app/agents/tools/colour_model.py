import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.core.config import settings


class ColourChangeInput(BaseModel):
    """Input schema for colour predictor requests."""

    red: float = Field(..., description="red scale percentage 0–5")
    green: float = Field(..., description="green scale percentage 0–5")
    blue: float = Field(..., description="blue scale percentage 0–5")
    salt: float = Field(..., description="salt g/L (40–80)")
    soda_ash: float = Field(..., description="soda ash g/L (10–20)")
    dyeing_temperature: float = Field(..., description="dyeing temp 60–80C")
    soaping_temperature: float = Field(..., description="soaping temp 70–95C")
    dyeing_time: int = Field(..., description="dyeing time 30–90 min")
    soaping_time: int = Field(..., description="soaping time 10–30 min")
    liquor_ratio: int = Field(..., description="liquor ratio 10–20")
    ph_level: float = Field(..., description="pH 10–11.5")
    water_hardness: int = Field(..., description="water hardness 50–300 ppm")


@tool("colour_predictor", args_schema=ColourChangeInput)
def colour_predictor(
    red: float,
    green: float,
    blue: float,
    salt: float,
    soda_ash: float,
    dyeing_temperature: float,
    soaping_temperature: float,
    dyeing_time: int,
    soaping_time: int,
    liquor_ratio: int,
    ph_level: float,
    water_hardness: int,
) -> dict:
    """
    Calls the external colour predictor API (your Google Colab deployed model).
    """

    print("--- TOOL CALLED: Predicting colour using ML model ---")

    api_url = f"{settings.CONVEX_BASE_URL}/colour/change"

    params = {
        "red": red,
        "green": green,
        "blue": blue,
        "salt": salt,
        "soda_ash": soda_ash,
        "dyeing_temperature": dyeing_temperature,
        "soaping_temperature": soaping_temperature,
        "dyeing_time": dyeing_time,
        "soaping_time": soaping_time,
        "liquor_ratio": liquor_ratio,
        "ph_level": ph_level,
        "water_hardness": water_hardness,
    }

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if "colour_change" not in data:
            return {
                "colour_change": False,
                "options": [],
                "error": "Invalid response format from API.",
            }

        return {
            "colour_change": True,
            "options": data["colour_change"],
        }

    except requests.exceptions.RequestException as e:
        return {"colour_change": False, "options": [], "error": f"API Error: {str(e)}"}

    except Exception as e:
        return {"colour_change": False, "options": [], "error": f"Unexpected error: {str(e)}"}
