from fastapi import FastAPI
from app.api.health import router as health_router

app = FastAPI(title="AI Power RAG Chatbot")

# Include API routers
app.include_router(health_router)
