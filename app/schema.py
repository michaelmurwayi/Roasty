from pydantic import BaseModel
from typing import List, Optional

class CoffeeLot(BaseModel):
    seller: str
    outturn: str
    grade: str
    bags: Optional[int]
    pockets: int = 0
    weight_kg: float  # changed to match your JSON

class FarmerDetails(BaseModel):
    name: str
    grower_code: str
    account_number: str
    bank: str
    branch: str
    swift_code: str = ""

class ExtractedSaleData(BaseModel):
    buyer: str
    coffee_details: List[CoffeeLot]  # changed to match your JSON
    # farmer_details: List[FarmerDetails]