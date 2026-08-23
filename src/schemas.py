from typing import List

from pydantic import BaseModel, Field


class ExtractedInfo(BaseModel):
    """Structured entities and summary pulled from a transcript.

    Replaces the previous free-text '- Topics: [list]' format: the LLM is
    now asked to return JSON matching this exact schema, which is parsed
    and validated instead of trusted as-is. If the model doesn't comply,
    extract_summary() raises LLMError rather than silently passing
    malformed text down the pipeline.
    """

    topics: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    quotes: List[str] = Field(default_factory=list)
    numbers_and_dates: List[str] = Field(default_factory=list)
    summary: str = ""


class FactCheckIssue(BaseModel):
    """A single discrepancy flagged between the article and its sources."""

    category: str
    description: str


class FactCheckReport(BaseModel):
    """Structured verdict from the verification stage.

    The key change from the original implementation: 'verdict' is now a
    machine-readable PASS/FAIL field that main.py can branch on (e.g. to
    set a non-zero exit code), rather than a report that is only ever
    printed and saved for a human to read later.
    """

    verdict: str  # "PASS" or "FAIL"
    issues: List[FactCheckIssue] = Field(default_factory=list)
