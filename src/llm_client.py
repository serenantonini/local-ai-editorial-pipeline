import logging
from typing import Optional

import ollama

from src.exceptions import LLMError

logger = logging.getLogger(__name__)


def chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.4,
    max_retries: int = 1,
) -> str:
    """Defensive wrapper around ollama.chat.

    Both editorial.py and verification.py previously called ollama.chat
    directly with near-identical boilerplate and no error handling. This
    consolidates that into one place and adds:

    - Actionable errors for the two most common failure modes: Ollama not
      running, and the requested model not being pulled locally.
    - A single retry (configurable) on empty responses or transient
      failures, since local LLMs occasionally return blank completions.
    - Optional JSON mode (Ollama's structured-output constraint) so
      callers that need a schema can request it directly.

    Raises LLMError on any unrecoverable failure. Callers should not need
    to catch ollama's own exception types.
    """
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
    }
    if json_mode:
        kwargs["format"] = "json"

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):
        try:
            response = ollama.chat(**kwargs)
        except Exception as e:  # ollama's client can raise several types; normalize them here
            last_error = e
            message = str(e).lower()
            if "connection" in message or "refused" in message:
                raise LLMError(
                    "Could not reach the local Ollama server. "
                    "Make sure it is running (`ollama serve`) and try again."
                ) from e
            if "not found" in message or "pull" in message:
                raise LLMError(
                    f"Model '{model}' is not available locally. "
                    f"Run `ollama pull {model}` and try again."
                ) from e
            logger.warning(
                "Ollama call attempt %d/%d raised an error: %s", attempt, max_retries + 1, e
            )
            continue

        content = response.get("message", {}).get("content", "")
        if content and content.strip():
            return content
        logger.warning(
            "Ollama returned an empty response on attempt %d/%d.", attempt, max_retries + 1
        )

    raise LLMError(
        f"LLM call to model '{model}' failed after {max_retries + 1} attempt(s)."
        + (f" Last error: {last_error}" if last_error else " The model kept returning empty responses.")
    )
