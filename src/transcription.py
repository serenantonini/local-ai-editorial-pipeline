import time
import whisper

def transcribe_audio(audio_path: str, model_size: str = "base", language: str = "it") -> str:
    """Loads local Whisper model and transcribes speech to text."""
    print(f"\n[1/4] Loading Whisper model ({model_size}) and transcribing ({language})...")
    model = whisper.load_model(model_size)
    start_time = time.time()
    
    # language=None triggers automatic language detection in Whisper
    lang_arg = None if language.lower() in ["auto", "none"] else language
    result = model.transcribe(audio_path, language=lang_arg, fp16=False)
    
    elapsed = round(time.time() - start_time, 2)
    print(f"Transcription completed in {elapsed}s.")
    return result["text"]