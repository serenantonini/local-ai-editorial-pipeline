import logging
import time
from pathlib import Path

import whisper

from src.exceptions import TranscriptionError

logger = logging.getLogger(__name__)

# Not an exhaustive allowlist (ffmpeg supports far more), just a sanity check
# to catch obviously wrong inputs (e.g. a .txt file) before loading the model.
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".webm"}


def _gpu_available() -> bool:
    """Detects whether fp16 (GPU) transcription can be used.

    The original code hardcoded fp16=False, which is safe but silently
    throws away a significant speedup on machines with a CUDA GPU. This
    keeps CPU-only machines working exactly as before while opting into
    fp16 automatically when a GPU is present.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def transcribe_audio(audio_path: str, model_size: str = "base", language: str = "it") -> str:
    """Loads a local Whisper model and transcribes speech to text.

    Raises TranscriptionError (instead of letting Whisper/ffmpeg exceptions
    propagate as raw tracebacks) if:
    - the audio file doesn't exist or isn't a file,
    - the Whisper model fails to load (e.g. invalid model_size),
    - transcription itself fails (e.g. corrupt/unsupported audio), or
    - transcription succeeds but returns no usable text (e.g. silent audio).
    """
    path = Path(audio_path)
    if not path.exists():
        raise TranscriptionError(f"Audio file not found: '{audio_path}'.")
    if not path.is_file():
        raise TranscriptionError(f"Expected a file, got a directory: '{audio_path}'.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "File extension '%s' is not in the commonly tested list %s. "
            "Whisper/ffmpeg may still handle it, but this is unverified.",
            path.suffix,
            sorted(SUPPORTED_EXTENSIONS),
        )

    print(f"\n[1/4] Loading Whisper model ({model_size}) and transcribing ({language})...")

    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to load Whisper model '{model_size}': {e}. "
            f"Valid sizes are: tiny, base, small, medium, large."
        ) from e

    use_fp16 = _gpu_available()
    # language=None triggers automatic language detection in Whisper
    lang_arg = None if language.lower() in {"auto", "none"} else language

    start_time = time.time()
    try:
        result = model.transcribe(str(path), language=lang_arg, fp16=use_fp16)
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed for '{audio_path}': {e}") from e
    elapsed = round(time.time() - start_time, 2)

    text = result.get("text", "").strip()
    if not text:
        raise TranscriptionError(
            f"Transcription produced no text for '{audio_path}'. "
            f"The audio may be silent, corrupted, or in an unsupported/mismatched "
            f"language (current setting: '{language}')."
        )

    print(f"Transcription completed in {elapsed}s ({'GPU/fp16' if use_fp16 else 'CPU/fp32'}).")
    return text
