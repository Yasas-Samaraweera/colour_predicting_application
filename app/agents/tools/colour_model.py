import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import settings

# Global variable to store the loaded model (lazy loading)
_loaded_model: Optional[object] = None
_model_path: Optional[Path] = None


def _get_model_path() -> Path:
    """Get the path to the trained model file."""
    # Get the absolute path to this file
    # File location: app/agents/tools/colour_model.py
    current_file = Path(__file__).resolve()
    
    # Start from the current file and go up to find the app directory
    # We know the structure: app/agents/tools/colour_model.py
    # So we need to go up 3 levels: tools -> agents -> app
    search_dir = current_file.parent  # Start at app/agents/tools/
    
    # Go up the directory tree until we find a directory containing 'ml_models'
    for _ in range(4):  # Max 4 levels up (tools -> agents -> app -> project_root)
        # Check if this directory or a subdirectory contains ml_models
        ml_models_dir = search_dir / "ml_models"
        if ml_models_dir.exists() and ml_models_dir.is_dir():
            return ml_models_dir / "colour_changing_predictor.pkl"
        
        # Also check if we're at project root and app/ml_models exists
        app_ml_models = search_dir / "app" / "ml_models"
        if app_ml_models.exists() and app_ml_models.is_dir():
            return app_ml_models / "colour_changing_predictor.pkl"
        
        # Move up one level
        if search_dir.parent == search_dir:  # Reached filesystem root
            break
        search_dir = search_dir.parent
    
    # Fallback: assume standard structure (go up 3 levels from current file)
    app_dir = current_file.parent.parent.parent
    model_path = app_dir / "ml_models" / "colour_changing_predictor.pkl"
    return model_path


def _load_model():
    """Load the trained model from the .pkl file (lazy loading)."""
    global _loaded_model, _model_path
    
    if _loaded_model is None:
        model_path = _get_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please ensure the trained model file 'colour_changing_predictor.pkl' exists."
            )
        
        try:
            _loaded_model = joblib.load(model_path)
            _model_path = model_path
            print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    return _loaded_model


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


def _prepare_model_input(
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
) -> pd.DataFrame:
    """
    Map tool input parameters to model input format.
    
    The model expects features in this exact order:
    dye_red_owf, dye_green_owf, dye_blue_owf, salt_gL, sodaAsh_gL, 
    temp_C, time_min, pH, liquor_ratio, water_hardness_ppm, 
    soap_temp_C, soap_time_min
    """
    model_input = pd.DataFrame({
        "dye_red_owf": [red],
        "dye_green_owf": [green],
        "dye_blue_owf": [blue],
        "salt_gL": [salt],
        "sodaAsh_gL": [soda_ash],
        "temp_C": [dyeing_temperature],
        "time_min": [dyeing_time],
        "pH": [ph_level],
        "liquor_ratio": [liquor_ratio],
        "water_hardness_ppm": [water_hardness],
        "soap_temp_C": [soaping_temperature],
        "soap_time_min": [soaping_time],
    })
    
    return model_input


def _make_prediction(
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
    Internal function to make RGB predictions using the loaded model.
    This is the core prediction logic shared by the tool and helper function.
    """
    try:
        # Load the model (lazy loading - only loads once)
        model = _load_model()
        
        # Prepare input data in the format expected by the model
        model_input = _prepare_model_input(
            red=red,
            green=green,
            blue=blue,
            salt=salt,
            soda_ash=soda_ash,
            dyeing_temperature=dyeing_temperature,
            soaping_temperature=soaping_temperature,
            dyeing_time=dyeing_time,
            soaping_time=soaping_time,
            liquor_ratio=liquor_ratio,
            ph_level=ph_level,
            water_hardness=water_hardness,
        )
        
        # Make prediction
        predicted_rgb = model.predict(model_input)
        
        # Extract RGB values (model returns array with shape [1, 3])
        rgb_r = int(predicted_rgb[0, 0])
        rgb_g = int(predicted_rgb[0, 1])
        rgb_b = int(predicted_rgb[0, 2])
        
        # Ensure RGB values are within valid range [0, 255]
        rgb_r = max(0, min(255, rgb_r))
        rgb_g = max(0, min(255, rgb_g))
        rgb_b = max(0, min(255, rgb_b))
        
        # Generate hex color code
        hex_color = f"#{rgb_r:02x}{rgb_g:02x}{rgb_b:02x}"
        
        return {
            "success": True,
            "predicted_rgb": {
                "R": rgb_r,
                "G": rgb_g,
                "B": rgb_b,
            },
            "hex_color": hex_color,
            "input_parameters": {
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
            },
        }

    except FileNotFoundError as e:
        error_msg = str(e)
        return {
            "success": False,
            "predicted_rgb": None,
            "hex_color": None,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        return {
            "success": False,
            "predicted_rgb": None,
            "hex_color": None,
            "error": error_msg,
        }


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
    Predicts RGB color values using the trained ML model based on dyeing parameters.
    
    This tool loads the local trained model and predicts RGB values (R, G, B) 
    based on the provided dyeing parameters.
    
    Returns:
        dict: A dictionary containing:
            - success: bool indicating if prediction was successful
            - predicted_rgb: dict with R, G, B values (0-255)
            - hex_color: hex color code (e.g., "#74cde7")
            - error: str error message if prediction failed
    """
    print("--- TOOL CALLED: Predicting colour using local ML model ---")
    
    result = _make_prediction(
        red=red,
        green=green,
        blue=blue,
        salt=salt,
        soda_ash=soda_ash,
        dyeing_temperature=dyeing_temperature,
        soaping_temperature=soaping_temperature,
        dyeing_time=dyeing_time,
        soaping_time=soaping_time,
        liquor_ratio=liquor_ratio,
        ph_level=ph_level,
        water_hardness=water_hardness,
    )
    
    if result.get("success"):
        rgb = result["predicted_rgb"]
        print(f"✓ Prediction successful: RGB({rgb['R']}, {rgb['G']}, {rgb['B']}) = {result['hex_color']}")
    else:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    return result


def predict_from_requirements(requirements: dict) -> dict:
    """
    Predict RGB color values from gathered requirements dictionary.
    
    This is a convenience function that extracts values from the requirements
    structure (as returned by requirements_graph) and calls the prediction model.
    
    Args:
        requirements: Dictionary containing the complete requirements structure
                     with keys: rgb_details, chemical_details, temperature_details,
                     time_details, liquor_ratio, pH, water_hardness_ppm
    
    Returns:
        dict: Prediction result with the same format as colour_predictor()
    
    Example:
        >>> requirements = {
        ...     "rgb_details": {"dye_red_owf": 2.5, "dye_green_owf": 1.0, "dye_blue_owf": 0.5},
        ...     "chemical_details": {"salt_gL": 50, "sodaAsh_gL": 15},
        ...     "temperature_details": {"temp_C": 70, "soap_temp_C": 80},
        ...     "time_details": {"time_min": 45, "soap_time_min": 15},
        ...     "liquor_ratio": {"liquor_ratio": 10},
        ...     "pH": {"pH": 10.5},
        ...     "water_hardness_ppm": {"water_hardness_ppm": 150}
        ... }
        >>> result = predict_from_requirements(requirements)
    """
    try:
        # Extract values from requirements structure
        rgb_details = requirements.get("rgb_details", {})
        chemical_details = requirements.get("chemical_details", {})
        temperature_details = requirements.get("temperature_details", {})
        time_details = requirements.get("time_details", {})
        liquor_ratio = requirements.get("liquor_ratio", {})
        ph_details = requirements.get("pH", {})
        water_hardness = requirements.get("water_hardness_ppm", {})
        
        # Call the core prediction function with extracted parameters
        return _make_prediction(
            red=rgb_details.get("dye_red_owf"),
            green=rgb_details.get("dye_green_owf"),
            blue=rgb_details.get("dye_blue_owf"),
            salt=chemical_details.get("salt_gL"),
            soda_ash=chemical_details.get("sodaAsh_gL"),
            dyeing_temperature=float(temperature_details.get("temp_C")),
            soaping_temperature=float(temperature_details.get("soap_temp_C")),
            dyeing_time=time_details.get("time_min"),
            soaping_time=time_details.get("soap_time_min"),
            liquor_ratio=int(liquor_ratio.get("liquor_ratio")),
            ph_level=ph_details.get("pH"),
            water_hardness=water_hardness.get("water_hardness_ppm"),
        )
        
    except KeyError as e:
        return {
            "success": False,
            "predicted_rgb": None,
            "hex_color": None,
            "error": f"Missing required field in requirements: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "predicted_rgb": None,
            "hex_color": None,
            "error": f"Failed to extract requirements: {str(e)}",
        }
