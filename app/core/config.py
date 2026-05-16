import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Production settings class"""
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "example_prod")
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
    
class DevSettings(Settings):
    """Development settings class"""
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "example_dev")

class TestSettings(Settings):
    """Test settings class"""
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "example_test")

@lru_cache
def get_settings():
    """Return settings based on ENV variable"""
    env = os.getenv("ENV", "dev")
    if env == "test":
        return TestSettings()
    if env == "dev":
        return DevSettings()
    return Settings()  # Default to production settings