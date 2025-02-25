from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    PROJECT_NAME: str = "Question Generator"
    
    class Config:
        env_file = ".env"

settings = Settings()
