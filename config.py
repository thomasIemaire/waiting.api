from dotenv import load_dotenv
from datetime import timedelta
import os

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()

    MONGO_URI = os.getenv("MONGO_URI")

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=int(os.getenv("JWT_EXPIRES_DAYS", "2")))

    JWT_TOKEN_LOCATION = ["headers"]
    JWT_HEADER_NAME = "Authorization"
    JWT_HEADER_TYPE = "Bearer"

    MAX_CONTENT_LENGTH = 1024 * 1024 * 24
    CELERY_BROKER_URL = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/0"
    PORT = int(os.getenv("PORT", 8888))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
