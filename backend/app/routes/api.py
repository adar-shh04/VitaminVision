from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from app.schemas.api_models import HealthResponse, PredictionResponse, HistoryResponse, PredictionHistoryItem
from app.services.ml_service import ml_service
from app.services.prediction_service import prediction_service
from app.db.database import get_db
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=not ml_service.is_demo_mode
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Business logic delegated entirely to Prediction Service
    result = await prediction_service.process(file, db)
    return PredictionResponse(**result)

@router.get("/history", response_model=HistoryResponse)
async def get_history(db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        collection = db.predictions
        # Get latest 50 predictions, sorted by newest first
        cursor = collection.find({}).sort("created_at", -1).limit(50)
        
        predictions = []
        async for doc in cursor:
            predictions.append(
                PredictionHistoryItem(
                    id=str(doc["_id"]),
                    filename=doc["filename"],
                    predicted_vitamin=doc["predicted_vitamin"],
                    confidence=doc["confidence"],
                    created_at=doc["created_at"]
                )
            )
            
        return HistoryResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Database read error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch history.")
