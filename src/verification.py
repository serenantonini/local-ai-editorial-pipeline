import ollama

def fact_check(transcript_text: str, summary_text: str, article_text: str, model: str = "llama3", target_language: str = "Italian") -> str:
    """Adversarial fact-checker comparing the generated article against source transcripts."""
    print(f"\n[4/4] Running anti-hallucination verification ({model})...")
    system_prompt = (
        f"You are a strict, adversarial editorial fact-checker. Your job is to compare the drafted article "
        f"against the original sources (transcript and extracted summary). "
        f"Identify ANY hallucination: fabricated or misspelled names, altered dates/numbers, unmentioned claims, "
        f"or falsely attributed quotes.\n\n"
        f"If the article is 100% faithful to the source, output: 'NO HALLUCINATIONS DETECTED'.\n"
        f"If discrepancies are found, list each issue with exact discrepancies between article and source.\n\n"
        f"Output your verdict and report in {target_language}."
    )
    user_prompt = (
        f"=== SOURCE 1: RAW TRANSCRIPT ===\n{transcript_text}\n\n"
        f"=== SOURCE 2: SUMMARY ===\n{summary_text}\n\n"
        f"=== ARTICLE UNDER REVIEW ===\n{article_text}\n\n"
        f"Execute verification and provide the fact-checking report in {target_language}."
    )
    response = ollama.chat(model=model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ])
    return response['message']['content']