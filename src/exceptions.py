class PipelineError(Exception):
    """Base exception for all pipeline-related errors.

    Catching this in main.py (instead of a bare Exception) lets us
    print a clean, user-facing error message and exit gracefully,
    while unexpected bugs still surface as normal tracebacks.
    """


class TranscriptionError(PipelineError):
    """Raised when audio transcription fails or produces unusable output.

    Covers: missing/unreadable audio files, Whisper model load failures,
    ffmpeg/decoding errors, and transcriptions that come back empty.
    """


class LLMError(PipelineError):
    """Raised when a local LLM call fails or returns an invalid response.

    Covers: Ollama server unreachable, model not pulled locally, and
    responses that don't match the schema the caller requested.
    """


class VerificationError(PipelineError):
    """Raised when the fact-check stage cannot produce a usable verdict.

    This is intentionally distinct from LLMError: a verification that
    the fact-checker itself flags as malformed should be treated as
    'this run must be reviewed manually', not just as a retryable
    generic LLM failure.
    """
