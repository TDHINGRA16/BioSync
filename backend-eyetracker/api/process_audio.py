import os
import logging
from io import BytesIO
from pathlib import Path
from typing import List

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from groq import Groq

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_api_keys() -> List[str]:
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        fallback_path = find_dotenv(usecwd=True)
        if fallback_path:
            load_dotenv(dotenv_path=fallback_path, override=False)
        else:
            logger.debug(".env file not found next to process_audio.py or in parent directories; relying on environment.")
    raw_keys = [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY_ALT_1"),
        os.getenv("GROQ_API_KEY_ALT_2"),
        os.getenv("GROQ_API_KEY_ALT_3"),
        os.getenv("GROQ_API_KEY_ALT_4"),
    ]
    keys = [key for key in raw_keys if key]
    if not keys:
        logger.error("No Groq API keys found. Check the .env file or environment variables.")
    return keys


GROQ_API_KEYS = _load_api_keys()

async def transcribe_audio(audio_file: UploadFile) -> str:
    audio_buffer = BytesIO(await audio_file.read())
    audio_buffer.name = audio_file.filename
    if not GROQ_API_KEYS:
        raise RuntimeError("No Groq API keys configured.")
    for api_key in GROQ_API_KEYS:
        client = Groq(api_key=api_key)
        try:
            logger.info(f"Trying transcription with key ending: {api_key[-6:]}")
            transcription = client.audio.transcriptions.create(
                file=(audio_buffer.name, audio_buffer.getvalue()),
                model="whisper-large-v3",
                response_format="json"
            )
            return transcription.text
        except Exception as e:
            logger.warning(f"Transcription failed with key ending {api_key[-6:]}, error: {e}")
            continue
    raise RuntimeError("All Groq API keys failed for transcription.")

@router.post("/audio/transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and get transcription as plain text.
    """
    try:
        text = await transcribe_audio(file)
        return JSONResponse(content={"text": text})
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
