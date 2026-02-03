from pydantic import BaseModel
from typing import List, Optional

class CoffeeLot(BaseModel):
    seller: str
    outturn: str
    grade: str
    bags: Optional[int]
    pockets: int = 0
    weight_kg: float  # changed to match your JSON

class ExtractedSaleData(BaseModel):
    buyer: str
    coffee_details: List[CoffeeLot]  # changed to match your JSON
