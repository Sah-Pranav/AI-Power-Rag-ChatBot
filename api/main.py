# api/main.py

"""Production FastAPI Application - Entry Point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from api.routes import health, query, documents
from utils.logger import logger

# Lifespan context manager (modern way)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 RAG API starting...")
    yield
    # Shutdown (if needed)
    logger.info("🛑 RAG API shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Chatbot API",
    description="Production RAG system with document Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Modern way
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(documents.router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )