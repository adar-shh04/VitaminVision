from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class VitaminInfo(BaseModel):
    icon: str
    color: str
    benefits: str
    sources: str
    daily_value: str

class PredictionResponse(BaseModel):
    predicted_vitamin: str
    confidence: float
    info: VitaminInfo
    is_demo_mode: bool

class PredictionHistoryItem(BaseModel):
    id: str
    filename: str
    predicted_vitamin: str
    confidence: float
    created_at: datetime

class HistoryResponse(BaseModel):
    predictions: List[PredictionHistoryItem]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
