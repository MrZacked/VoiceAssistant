import os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .asr import initialize_asr, transcribe_audio_bytes
from .llm import initialize_llm, generate_assistant_response
from .tts import is_tts_enabled, synthesize_to_file

APP_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_ROOT / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESPONSES_DIR = STATIC_DIR / "responses"


def ensure_static_dir() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Voice Assistant API")

ensure_static_dir()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    initialize_asr()
    initialize_llm()



@app.post("/ask_assistant")
async def ask_assistant(background_tasks: BackgroundTasks, audio_file: UploadFile = File(...)):
    if audio_file is None:
        raise HTTPException(status_code=400, detail="Missing 'audio_file' in form-data")

    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Save input audio for debugging (optional)
    orig_ext = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
    orig_name = f"input_{uuid4().hex}{orig_ext}"
    orig_path = UPLOADS_DIR / orig_name
    try:
        with open(orig_path, "wb") as f:
            f.write(audio_bytes)
    except Exception:
        pass

    try:
        transcribed_text = transcribe_audio_bytes(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

    try:
        assistant_response = generate_assistant_response(transcribed_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")

    response_audio_url = None
    if is_tts_enabled() and assistant_response:
        filename = f"response_{uuid4().hex}.wav"
        out_path = RESPONSES_DIR / filename
        try:
            # Generate audio synchronously for now to ensure it works
            audio_result = synthesize_to_file(assistant_response, str(out_path))
            if audio_result and out_path.exists():
                response_audio_url = f"/responses/{filename}"
        except Exception as e:
            print(f"TTS failed: {e}")
            response_audio_url = None

    return JSONResponse(
        content={
            "transcribed_text": transcribed_text,
            "assistant_response": assistant_response,
            "response_audio_url": response_audio_url,
        }
    )





