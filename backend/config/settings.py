from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    BUCKET_NAME: str
    HUGGING_FACE_API_KEY: str
    MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.2"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()