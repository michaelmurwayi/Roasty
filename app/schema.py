from pydantic import BaseModel
from typing import List, Optional

class CoffeeLot(BaseModel):
    seller: str
    outturn: str
    grade: str
    bags: Optional[int]
    pockets: Optional[int]
    weight: float


class ExtractedSaleData(BaseModel):
    buyer: str
    coffee: List[CoffeeLot]