from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import api
from app.services.ml_service import ml_service
from app.db.database import connect_to_mongo, close_mongo_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    await connect_to_mongo()
    ml_service.load_model()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    await close_mongo_connection()

app = FastAPI(
    title="Vitamin Vision API",
    description="Backend API for the Vitamin Vision ML Application",
    version="1.0.0",
    lifespan=lifespan
)

# Allow Streamlit frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router, prefix="/api/v1")
