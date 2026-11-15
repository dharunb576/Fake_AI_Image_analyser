
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import API_TITLE, API_VERSION
from app.services.predictor import predictor

from app.api.routes import detection, health, model_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model once when the API starts
    print("Loading AI Fake Photo Detector model...")
    predictor.load_model()
    print("Backend is ready!")

    yield

    # Application shutdown
    print("Backend shutting down...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="API for detecting AI-generated and real images.",
    lifespan=lifespan
)


# Allow the React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes
app.include_router(detection.router)
app.include_router(health.router)
app.include_router(model_info.router)


@app.get("/")
def root():
    return {
        "message": "AI Fake Photo Detector API is running!",
        "documentation": "/docs",
        "health": "/health"
    }
