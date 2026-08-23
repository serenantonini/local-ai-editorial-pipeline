import json
import logging

from pydantic import ValidationError

from src.exceptions import VerificationError
from src.llm_client import chat
from src.schemas import ExtractedInfo, FactCheckReport

logger = logging.getLogger(__name__)


def fact_check(
    transcript_text: str,
    extracted: ExtractedInfo,
    article_text: str,
    model: str = "llama3",
    target_language: str = "Italian",
) -> FactCheckReport:
    """Adversarial fact-checker comparing the generated article against source material.

    Returns a validated FactCheckReport with an explicit "PASS"/"FAIL"
    verdict and a structured list of issues, instead of a free-text report
    that only a human can act on. main.py uses report.verdict to decide the
    process exit code and to flag runs that need manual review before
    publishing.

    Note: as documented in the project README, this remains an AI-assisted
    consistency check between the article and the transcript/extraction —
    not independent, source-grounded fact verification.
    """
    print(f"\n[4/4] Running anti-hallucination verification ({model})...")

    schema_hint = (
        '{"verdict": "PASS" or "FAIL", '
        '"issues": [{"category": "...", "description": "..."}]}'
    )
    system_prompt = (
        "You are a strict, adversarial editorial fact-checker. Compare the "
        "drafted article against the original sources (transcript and "
        "extracted info) and identify ANY hallucination: fabricated or "
        "misspelled names, altered dates/numbers, unmentioned claims, or "
        "falsely attributed quotes.\n\n"
        "Return ONLY a valid JSON object matching this exact schema, with no "
        f"surrounding prose and no markdown code fences:\n{schema_hint}\n\n"
        'Use verdict "PASS" only if there are zero discrepancies; otherwise '
        'use "FAIL" and list every issue found. '
        f"All 'description' text must be written in {target_language}."
    )
    user_prompt = (
        f"=== SOURCE 1: RAW TRANSCRIPT ===\n{transcript_text}\n\n"
        f"=== SOURCE 2: EXTRACTED INFO (JSON) ===\n{extracted.model_dump_json(indent=2)}\n\n"
        f"=== ARTICLE UNDER REVIEW ===\n{article_text}\n\n"
        "Execute verification now."
    )

    raw = chat(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        temperature=0.1,  # a fact-checker should be as deterministic as possible
    )

    try:
        data = json.loads(raw)
        report = FactCheckReport(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise VerificationError(
            "The fact-checker returned a response that could not be parsed "
            f"as a valid verdict: {e}\nRaw output (truncated):\n{raw[:500]}\n"
            "Treat this run as UNVERIFIED and review the article manually."
        ) from e

    if report.verdict.upper() not in {"PASS", "FAIL"}:
        raise VerificationError(
            f"Fact-checker returned an unexpected verdict value: '{report.verdict}'. "
            "Expected 'PASS' or 'FAIL'."
        )

    return report
