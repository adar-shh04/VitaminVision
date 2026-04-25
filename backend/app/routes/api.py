from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from app.schemas.api_models import HealthResponse, PredictionResponse, HistoryResponse, PredictionHistoryItem
from app.services.ml_service import ml_service
from app.db.database import get_database
from app.models.domain import PredictionRecord
from PIL import Image
import io
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
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Run inference
        result = ml_service.predict(image)
        
        # Save to database
        db = get_database()
        record = PredictionRecord(
            filename=file.filename,
            predicted_vitamin=result["predicted_vitamin"],
            confidence=result["confidence"]
        )
        
        # Insert into MongoDB collection
        collection = db.predictions
        new_record = await collection.insert_one(record.model_dump(by_alias=True, exclude=["id"]))
        
    except Exception as e:
        logger.error(f"Prediction or database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

    return PredictionResponse(**result)

@router.get("/history", response_model=HistoryResponse)
async def get_history():
    try:
        db = get_database()
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
        logger.error(f"Database read error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history.")
