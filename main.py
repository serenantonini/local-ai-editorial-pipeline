import argparse
from pathlib import Path
from src.transcription import transcribe_audio
from src.editorial import extract_summary, generate_article
from src.verification import fact_check

def run_pipeline(
    audio_path: str,
    model_llm: str = "llama3",
    whisper_size: str = "base",
    audio_lang: str = "it",
    target_lang: str = "Italian",
    output_dir: str = "output"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Local Transcription
    transcript = transcribe_audio(audio_path, model_size=whisper_size, language=audio_lang)
    (out_path / "1_transcript.txt").write_text(transcript, encoding="utf-8")
    
    # 2. Information Extraction & Summary
    summary = extract_summary(transcript, model=model_llm, target_language=target_lang)
    (out_path / "2_summary.txt").write_text(summary, encoding="utf-8")
    
    # 3. Article Drafting
    article = generate_article(transcript, summary, model=model_llm, target_language=target_lang)
    (out_path / "3_article.md").write_text(article, encoding="utf-8")
    
    # 4. Anti-Hallucination Verification
    check_report = fact_check(transcript, summary, article, model=model_llm, target_language=target_lang)
    (out_path / "4_fact_check.txt").write_text(check_report, encoding="utf-8")
    
    print(f"\nPipeline successfully completed! Artifacts saved to '{output_dir}/'")
    print(f"\n--- FACT-CHECK REPORT ---\n{check_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local AI Audio Transcription, Editorial & Verification Pipeline")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.mp3, .wav, .m4a)")
    parser.add_argument("--llm", type=str, default="llama3", help="Local Ollama model (default: llama3)")
    parser.add_argument("--whisper", type=str, default="base", help="Whisper model size: tiny, base, small, medium, large (default: base)")
    parser.add_argument("--audio-lang", type=str, default="it", help="Spoken language code for Whisper, or 'auto' (default: it)")
    parser.add_argument("--target-lang", type=str, default="Italian", help="Language for generated text outputs (default: Italian)")
    parser.add_argument("--out", type=str, default="output", help="Directory where artifacts are saved (default: output)")
    
    args = parser.parse_args()
    run_pipeline(
        audio_path=args.audio,
        model_llm=args.llm,
        whisper_size=args.whisper,
        audio_lang=args.audio_lang,
        target_lang=args.target_lang,
        output_dir=args.out
    )