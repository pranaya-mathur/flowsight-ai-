"""FastAPI application for FlowSight predictions."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import router, initialize_predictor
from src.logging_utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    # Startup
    logger.info("Starting FlowSight API...")
    initialize_predictor()
    logger.info("FlowSight API ready")
    yield
    # Shutdown
    logger.info("Shutting down FlowSight API...")


# Create FastAPI app
app = FastAPI(
    title="FlowSight AI",
    description="AI-Powered Supply Chain Delay Prediction System",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, tags=["predictions"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
