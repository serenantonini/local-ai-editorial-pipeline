import argparse
import logging
import sys
from pathlib import Path

from src.editorial import extract_summary, generate_article
from src.exceptions import PipelineError
from src.transcription import transcribe_audio
from src.verification import fact_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Exit codes: distinct from "crashed" so scripts/CI calling this pipeline
# can tell "the pipeline itself broke" apart from "it ran fine, but the
# fact-checker flagged the result and a human should look at it".
EXIT_OK = 0
EXIT_PIPELINE_ERROR = 1
EXIT_FACT_CHECK_FAILED = 2


def run_pipeline(
    audio_path: str,
    model_llm: str = "llama3",
    whisper_size: str = "base",
    audio_lang: str = "it",
    target_lang: str = "Italian",
    output_dir: str = "output",
) -> int:
    """Runs the full transcription -> extraction -> drafting -> verification pipeline.

    Returns a process exit code:
    - EXIT_OK (0): completed, fact-check verdict was PASS.
    - EXIT_PIPELINE_ERROR (1): a stage failed (bad audio, Ollama unreachable,
      malformed model output, etc.). See the logged error for details.
    - EXIT_FACT_CHECK_FAILED (2): the pipeline completed and all artifacts
      were saved, but the fact-checker flagged discrepancies -- the article
      needs manual review before publishing.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Local Transcription
        transcript = transcribe_audio(audio_path, model_size=whisper_size, language=audio_lang)
        (out_path / "1_transcript.txt").write_text(transcript, encoding="utf-8")

        # 2. Information Extraction & Summary (structured, validated JSON)
        extracted = extract_summary(transcript, model=model_llm, target_language=target_lang)
        (out_path / "2_summary.json").write_text(extracted.model_dump_json(indent=2), encoding="utf-8")

        # 3. Article Drafting
        article = generate_article(transcript, extracted, model=model_llm, target_language=target_lang)
        (out_path / "3_article.md").write_text(article, encoding="utf-8")

        # 4. Anti-Hallucination Verification (structured PASS/FAIL verdict)
        report = fact_check(transcript, extracted, article, model=model_llm, target_language=target_lang)
        (out_path / "4_fact_check.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    except PipelineError as e:
        # Any of the four stages raises a typed PipelineError on failure;
        # we stop here instead of letting a partial/broken pipeline continue.
        logger.error("Pipeline stopped: %s", e)
        return EXIT_PIPELINE_ERROR

    print(f"\nPipeline completed. Artifacts saved to '{output_dir}/'")
    print(f"\n--- FACT-CHECK VERDICT: {report.verdict.upper()} ---")

    if report.issues:
        for issue in report.issues:
            print(f"  - [{issue.category}] {issue.description}")
        logger.warning(
            "The fact-checker flagged %d issue(s). Review '%s' before publishing.",
            len(report.issues),
            out_path / "3_article.md",
        )
        return EXIT_FACT_CHECK_FAILED

    return EXIT_OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local AI Audio Transcription, Editorial & Verification Pipeline")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.mp3, .wav, .m4a)")
    parser.add_argument("--llm", type=str, default="llama3", help="Local Ollama model (default: llama3)")
    parser.add_argument(
        "--whisper", type=str, default="base",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    parser.add_argument("--audio-lang", type=str, default="it", help="Spoken language code for Whisper, or 'auto' (default: it)")
    parser.add_argument("--target-lang", type=str, default="Italian", help="Language for generated text outputs (default: Italian)")
    parser.add_argument("--out", type=str, default="output", help="Directory where artifacts are saved (default: output)")

    args = parser.parse_args()
    exit_code = run_pipeline(
        audio_path=args.audio,
        model_llm=args.llm,
        whisper_size=args.whisper,
        audio_lang=args.audio_lang,
        target_lang=args.target_lang,
        output_dir=args.out,
    )
    sys.exit(exit_code)
