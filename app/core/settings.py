from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AI Power RAG Chatbot"
    version: str = "0.1.0"
    environment: str = "local"

settings = Settings()
