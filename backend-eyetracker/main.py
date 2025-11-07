from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.process_audio import router as audio_router


# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Alice API",
    description="Alice - Search, audio transcription, computer control, bot control",
    version="2.0.0",
)

# Allow all origins for testing/dev, adjust as needed for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(audio_router, prefix="", tags=["audio"])

@app.get("/")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    """Initialize bot MQTT client and conversation context on startup"""
    logger.info("🤖 Starting Alice AI Backend...")
    logger.info("🚀 Alice AI Backend ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup bot MQTT client on shutdown"""
    logger.info("👋 Goodbye!")

