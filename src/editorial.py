import json
import logging

from pydantic import ValidationError

from src.exceptions import LLMError
from src.llm_client import chat
from src.schemas import ExtractedInfo

logger = logging.getLogger(__name__)

# Conservative character budget for a single prompt. llama3's default context
# window in Ollama is commonly 8K tokens; ~24K characters (~6K tokens) leaves
# headroom for the system prompt, JSON schema instructions, and the model's
# own response. This is a guard against *silent* truncation/degradation by
# Ollama, not a proper long-document strategy (see README "Future Work").
MAX_TRANSCRIPT_CHARS = 24_000


def _truncate_if_needed(text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    logger.warning(
        "Transcript is %d characters, exceeding the %d-character safety limit. "
        "Truncating to reduce the risk of silent context overflow; long "
        "recordings should eventually be chunked instead (see Future Work).",
        len(text),
        max_chars,
    )
    return text[:max_chars]


def extract_summary(transcript_text: str, model: str = "llama3", target_language: str = "Italian") -> ExtractedInfo:
    """Extracts structured entities and an analytical summary.

    Returns a validated ExtractedInfo instead of a free-text string: the
    model is asked for JSON matching an explicit schema, and the response
    is parsed and validated with Pydantic. If the model doesn't comply,
    this raises LLMError with the raw output attached, rather than passing
    malformed text silently down to article generation and fact-checking.
    """
    print(f"\n[2/4] Extracting structured information and summary ({model})...")

    transcript_text = _truncate_if_needed(transcript_text)

    schema_hint = (
        '{"topics": [...], "people": [...], "locations": [...], '
        '"key_facts": [...], "quotes": [...], "numbers_and_dates": [...], '
        '"summary": "..."}'
    )
    system_prompt = (
        "You are an analytical editorial assistant. Read the transcript and "
        "return ONLY a valid JSON object matching this exact schema, with no "
        f"surrounding prose and no markdown code fences:\n{schema_hint}\n\n"
        f"All string values must be written in {target_language}. "
        "Every list key must be present even if empty; 'summary' must be a "
        "concise, thorough paragraph covering all main points."
    )
    user_prompt = f"Full Transcript:\n\n{transcript_text}"

    raw = chat(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        temperature=0.2,  # lower temperature: this is extraction, not creative writing
    )

    try:
        data = json.loads(raw)
        return ExtractedInfo(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise LLMError(
            "The information-extraction step returned output that could not "
            f"be parsed as valid JSON matching the expected schema: {e}\n"
            f"Raw output (truncated):\n{raw[:500]}"
        ) from e


def generate_article(
    transcript_text: str,
    extracted: ExtractedInfo,
    model: str = "llama3",
    target_language: str = "Italian",
) -> str:
    """Drafts an engaging SEO/web article in Markdown format in the target language."""
    print(f"\n[3/4] Drafting publication article ({model})...")

    transcript_text = _truncate_if_needed(transcript_text)
    extracted_block = extracted.model_dump_json(indent=2)

    system_prompt = (
        "You are an expert web copywriter. Draft a publication-ready article "
        "based on the episode. Use the EXTRACTED INFORMATION for key data "
        "points and the FULL TRANSCRIPT for context and tone. Format with "
        "clear Markdown headings (H1, H2) and well-structured paragraphs. "
        f"Important: write the entire article in {target_language}."
    )
    user_prompt = (
        f"=== EXTRACTED INFO & SUMMARY (JSON) ===\n{extracted_block}\n\n"
        f"=== FULL TRANSCRIPT ===\n{transcript_text}\n\n"
        f"Draft the full article now in {target_language}."
    )
    return chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.6)
