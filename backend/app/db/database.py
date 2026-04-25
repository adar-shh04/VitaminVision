from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "vitamin_vision")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../models/my_model.h5")

settings = Settings()

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.MONGODB_URI)
    print(f"Connected to MongoDB at {settings.MONGODB_URI}")

async def close_mongo_connection():
    if db.client is not None:
        db.client.close()
        print("Closed MongoDB connection.")

async def get_db():
    """Dependency injection for FastAPI routes."""
    return db.client[settings.DATABASE_NAME]
