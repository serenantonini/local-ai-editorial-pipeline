import ollama

def extract_summary(transcript_text: str, model: str = "llama3", target_language: str = "Italian") -> str:
    """Extracts structured entities and generates an analytical summary in the target language."""
    print(f"\n[2/4] Extracting structured information and summary ({model})...")
    system_prompt = (
        f"You are an analytical editorial assistant. Your task is to process the provided audio transcript "
        f"and return a clean, structured output in {target_language} with exactly two sections:\n\n"
        "1. EXTRACTED INFORMATION:\n"
        "- Topics: [list]\n"
        "- People: [list]\n"
        "- Locations: [list]\n"
        "- Key Facts: [list]\n"
        "- Quotes: [list]\n"
        "- Numbers / Dates: [list]\n\n"
        "2. EPISODE SUMMARY:\n"
        "Write a concise, thorough summary covering all main points.\n\n"
        f"Important: Ensure all output content is written in {target_language}."
    )
    response = ollama.chat(model=model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Full Transcript:\n\n{transcript_text}"}
    ])
    return response['message']['content']

def generate_article(transcript_text: str, summary_text: str, model: str = "llama3", target_language: str = "Italian") -> str:
    """Drafts an engaging SEO/web article in Markdown format in the target language."""
    print(f"\n[3/4] Drafting publication article ({model})...")
    system_prompt = (
        f"You are an expert web copywriter. Draft a publication-ready article based on the episode. "
        f"Use the EXTRACTED INFORMATION for key data points and the FULL TRANSCRIPT for context and tone. "
        f"Format with clear Markdown headings (H1, H2) and well-structured paragraphs. "
        f"Important: Write the entire article in {target_language}."
    )
    user_prompt = (
        f"=== EXTRACTED INFO & SUMMARY ===\n{summary_text}\n\n"
        f"=== FULL TRANSCRIPT ===\n{transcript_text}\n\n"
        f"Draft the full article now in {target_language}."
    )
    response = ollama.chat(model=model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ])
    return response['message']['content']