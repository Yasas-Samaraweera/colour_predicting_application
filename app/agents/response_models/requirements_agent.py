from pydantic import BaseModel, Field
from typing import List, Optional


class RgbDetails(BaseModel):
    dye_red_owf: float = Field(..., description="red scale as a percentage between 0% and 5%")
    dye_green_owf: float = Field(..., description="green scale as a percentage between 0% and 5%")
    dye_blue_owf: float = Field(..., description="blue scale as a percentage between 0% and 5%")


class ChemicalDetails(BaseModel):
    salt_gL: float = Field(..., description="salt concentration between 40–80 g/L")
    sodaAsh_gL: float = Field(..., description="soda ash concentration between 10–20 g/L")


class TemperatureDetails(BaseModel):
    temp_C: int = Field(..., description="Dyeing temperature 60–80°C")
    soap_temp_C: int = Field(..., description="Soaping temperature 70–95°C")


class TimeDetails(BaseModel):
    time_min: int = Field(..., description="Dyeing time 30–90 min")
    soap_time_min: int = Field(..., description="Soaping time 10–30 min")


class LiquorRatio(BaseModel):
    liquor_ratio: float = Field(..., description="Water ratio for 1kg of fabric (10–20)")


class PhDetails(BaseModel):
    pH: float = Field(..., description="pH of water between 10 and 11.5")


class WaterHardness(BaseModel):
    water_hardness_ppm : int = Field(..., description="Water hardness 50–300 ppm")


class UserConfirmations(BaseModel):
    accept_outbound_top_option: bool = Field(...)
    notes: Optional[str] = Field(None)


class MissingInfo(BaseModel):
    missing_fields: List[str] = Field(...)
    question: str = Field(...)


class CompleteRequirements(BaseModel):
    rgb_details: RgbDetails
    chemical_details: ChemicalDetails
    temperature_details: TemperatureDetails
    time_details: TimeDetails
    liquor_ratio: LiquorRatio
    pH: PhDetails
    water_hardness_ppm : WaterHardness
    user_confirmations: UserConfirmations
    missing_info: MissingInfo


class RequirementsResponseModel(BaseModel):
    requirements: CompleteRequirements

