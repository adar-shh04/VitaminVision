import logging
from PIL import Image
import io
from fastapi import UploadFile, HTTPException
from app.services.ml_service import ml_service
from app.models.domain import PredictionRecord
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class PredictionService:
    async def process(self, file: UploadFile, db: AsyncIOMotorDatabase) -> dict:
        """Coordinates the prediction workflow: preprocessing -> inference -> db persistence."""
        logger.info(f"Processing new prediction request for file: {file.filename}")
        
        # 1. Preprocessing
        try:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            logger.error(f"Image preprocessing failed for {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid image file provided.")

        # 2. ML Inference
        try:
            result = ml_service.predict(image)
        except Exception as e:
            logger.error(f"ML inference failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during model inference.")

        # 3. Database Persistence
        try:
            record = PredictionRecord(
                filename=file.filename,
                predicted_vitamin=result["predicted_vitamin"],
                confidence=result["confidence"]
            )
            collection = db.predictions
            await collection.insert_one(record.model_dump(by_alias=True, exclude=["id"]))
            logger.info("Successfully persisted prediction record to MongoDB.")
        except Exception as e:
            logger.error(f"Database insertion failed: {e}", exc_info=True)
            # We don't raise here, we can still return the prediction even if logging fails.
            # But in some systems, you might want to raise.

        return result

prediction_service = PredictionService()
