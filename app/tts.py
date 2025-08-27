import os
from pathlib import Path
from typing import Optional

import pyttsx3
import ffmpeg

_engine: Optional[pyttsx3.Engine] = None
_coqui_tts = None


def is_tts_enabled() -> bool:
    return os.getenv("ENABLE_TTS", "1") == "1"


def _initialize_pyttsx3() -> None:
    global _engine
    if _engine is not None:
        return
    _engine = pyttsx3.init()
    rate = int(os.getenv("TTS_RATE", "175"))
    volume = float(os.getenv("TTS_VOLUME", "1.0"))
    _engine.setProperty("rate", rate)
    _engine.setProperty("volume", volume)
    
    voices = _engine.getProperty('voices')
    turkish_voices = [v for v in voices if 'turkish' in v.name.lower() or 'tr' in v.id.lower()]
    if turkish_voices:
        _engine.setProperty('voice', turkish_voices[0].id)
    else:
        if voices:
            _engine.setProperty('voice', voices[0].id)


def _initialize_coqui() -> None:
    global _coqui_tts
    if _coqui_tts is not None:
        return
    try:
        from TTS.api import TTS  # type: ignore
    except Exception:
        _coqui_tts = None
        return
    model_name = os.getenv("COQUI_TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
    use_gpu = os.getenv("FORCE_CUDA", "0") == "1"
    try:
        print("Loading Coqui TTS model...")
        _coqui_tts = TTS(model_name=model_name, gpu=use_gpu)
        print("Coqui TTS model loaded successfully")
    except Exception as e:
        print(f"Failed to load Coqui TTS: {e}")
        _coqui_tts = None


def detect_turkish(text: str) -> bool:
    if not text:
        return False
    
    turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']
    if any(char in text for char in turkish_chars):
        return True
    
    turkish_words = [
        'merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar',
        'sipariş', 'ürün', 'iade', 'kargo', 'ödeme', 'hesap', 'yardım',
        'nerede', 'ne', 'nasıl', 'neden', 'ne zaman', 'hangi', 'kim',
        'evet', 'hayır', 'teşekkür', 'lütfen', 'rica', 'tamam'
    ]
    
    text_lower = text.lower()
    return any(word in text_lower for word in turkish_words)


def synthesize_to_file(text: str, file_path: str) -> Optional[str]:
    if not is_tts_enabled():
        return None
    
    is_turkish = detect_turkish(text)
    prefer = os.getenv("TTS_ENGINE", "coqui").lower()
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer == "coqui":
        _initialize_coqui()
        if _coqui_tts is not None:
            try:
                if is_turkish:
                    try:
                        from TTS.api import TTS
                        turkish_tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", gpu=False)
                        turkish_tts.tts_to_file(text=text, file_path=str(out_path))
                        return str(out_path)
                    except Exception:
                        pass
                
                _coqui_tts.tts_to_file(text=text, file_path=str(out_path))
                return str(out_path)
            except Exception:
                pass

    # Use pyttsx3 as fallback
    _initialize_pyttsx3()
    
    if is_turkish:
        voices = _engine.getProperty('voices')
        turkish_voices = [v for v in voices if 'turkish' in v.name.lower() or 'tr' in v.id.lower()]
        if turkish_voices:
            _engine.setProperty('voice', turkish_voices[0].id)
    
    _engine.save_to_file(text, str(out_path))
    _engine.runAndWait()
    try:
        if out_path.exists() and out_path.stat().st_size > 44:
            return str(out_path)
        return None
    except Exception:
        return None


def convert_wav_to_mp3(wav_path: str, mp3_path: str) -> Optional[str]:
    try:
        (
            ffmpeg
            .input(wav_path)
            .output(mp3_path, acodec="libmp3lame", audio_bitrate="192k")
            .overwrite_output()
            .run(quiet=True)
        )
        out = Path(mp3_path)
        return str(out) if out.exists() and out.stat().st_size > 0 else None
    except Exception:
        return None

