import os
import tempfile
import warnings
from typing import Optional

import whisper
import torch

warnings.filterwarnings("ignore", message=".*CUDA capability.*")


_whisper_model: Optional[whisper.Whisper] = None


def initialize_asr() -> None:
    global _whisper_model
    if _whisper_model is not None:
        return
    model_name = os.getenv("WHISPER_MODEL", "base")
    
    print("Loading Whisper model...")
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 8:
                print("Loading Whisper model on CUDA with optimized settings...")
                _whisper_model = whisper.load_model(model_name, device="cuda")
            else:
                print("Loading Whisper model on CUDA...")
                _whisper_model = whisper.load_model(model_name, device="cuda")
            print("Whisper model loaded on CUDA successfully")
        else:
            print("Loading Whisper model on CPU...")
            _whisper_model = whisper.load_model(model_name, device="cpu")
            print("Whisper model loaded on CPU successfully")
    except Exception as e:
        print(f"CUDA failed, falling back to CPU: {e}")
        _whisper_model = whisper.load_model(model_name, device="cpu")
        print("Whisper model loaded on CPU successfully")


def transcribe_audio_bytes(audio_bytes: bytes, language: Optional[str] = None) -> str:
    if _whisper_model is None:
        initialize_asr()
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name
        
        if language is None:
            result = _whisper_model.transcribe(tmp_path, language=None)
        else:
            result = _whisper_model.transcribe(tmp_path, language=language)
        
        text = (result.get("text") or "").strip()
        detected_lang = result.get("language", "en")
        
        if detected_lang == "tr":
            print(f"Turkish detected: {text}")
        
        return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

