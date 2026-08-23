# Local AI Editorial Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Whisper](https://img.shields.io/badge/Whisper-Speech--to--Text-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An **offline, privacy-first AI pipeline** for transforming audio recordings into structured editorial content using local speech recognition and Large Language Models.

The pipeline automatically performs:

**audio transcription → information extraction → article generation → AI-assisted verification**

All language-model inference is performed locally using **OpenAI Whisper** and **Ollama**, avoiding the need to send audio or generated content to external AI APIs.

---

## Project Overview

The goal of this project is to explore how locally running AI models can support an editorial workflow while keeping the underlying content on the user's machine.

Given an audio recording, the system:

1. Transcribes the audio using **Whisper**.
2. Extracts and summarizes the main information using a local LLM.
3. Generates a structured article from the transcript and summary.
4. Performs a separate verification pass to identify potential inconsistencies or unsupported claims.

The project is designed as a modular pipeline rather than a single prompt-based application.

---

## Pipeline

```text
              Audio File
                  │
                  ▼
        ┌───────────────────┐
        │      Whisper      │
        │  Speech-to-Text   │
        └─────────┬─────────┘
                  │
                  ▼
             Transcript
                  │
                  ▼
        ┌───────────────────┐
        │      Ollama       │
        │     Local LLM     │
        └─────────┬─────────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
     Information       Article
      Extraction       Generation
          │                │
          └───────┬────────┘
                  ▼
        ┌───────────────────┐
        │    Verification   │
        │    / Fact-Check   │
        └─────────┬─────────┘
                  │
                  ▼
             Final Report
```

---

## Architecture

The pipeline is divided into three main modules:

| Module | Responsibility |
|---|---|
| `transcription.py` | Converts audio into text using Whisper |
| `editorial.py` | Extracts information and generates the article using a local LLM |
| `verification.py` | Performs an AI-assisted verification pass on the generated content |

The orchestration logic is contained in `main.py`.

The pipeline produces four intermediate/final artifacts:

```text
output/
├── 1_transcript.txt
├── 2_summary.txt
├── 3_article.md
└── 4_fact_check.txt
```

This makes each stage independently inspectable rather than treating the generated article as a black box.

---

## Technologies

- **Python**
- **OpenAI Whisper** — local speech-to-text
- **Ollama** — local LLM inference
- **Llama 3** — default language model
- **PyTorch / Torchaudio** — audio and model dependencies

---

## Getting Started

### Prerequisites

Install:

- Python 3.10+
- [Ollama](https://ollama.com/)
- A local Ollama model, such as `llama3`

Pull the default model:

```bash
ollama pull llama3
```

### Installation

Clone the repository:

```bash
git clone https://github.com/serenantonini/local-ai-editorial-pipeline.git
cd local-ai-editorial-pipeline
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the pipeline by providing an audio file:

```bash
python main.py --audio path/to/audio.mp3
```

By default, the pipeline uses:

```text
LLM:          llama3
Whisper:      base
Audio:        Italian
Output:       output/
```

These parameters can be customized:

```bash
python main.py \
    --audio path/to/audio.mp3 \
    --llm llama3 \
    --whisper base \
    --audio-lang it \
    --target-lang Italian \
    --out output
```

Supported Whisper model sizes include:

```text
tiny
base
small
medium
large
```

---

## Repository Structure

```text
local-ai-editorial-pipeline/
│
├── src/
│   ├── __init__.py
│   ├── transcription.py
│   ├── editorial.py
│   └── verification.py
│
├── main.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Privacy & Local Processing

A central design goal of the project is **local processing**.

Audio transcription is performed locally through Whisper, while text generation and verification are handled by a locally running LLM through Ollama.

This architecture can be useful for workflows involving sensitive recordings where sending raw audio or transcripts to external AI services may be undesirable.

> Local inference does not automatically guarantee privacy or security; the operating environment and locally stored artifacts must also be protected appropriately.

---

## Limitations

The verification stage should be considered an **AI-assisted consistency check**, not a fully automated fact-checking system.

In particular:

- LLMs can generate or overlook factual errors.
- Verification performed by another LLM does not constitute independent source verification.
- Whisper transcription quality depends on audio quality, speakers, accents, and background noise.
- Local inference can require substantial computational resources depending on the selected models.
- The current pipeline does not automatically retrieve or cross-reference external authoritative sources.

The system should therefore be treated as an **editorial assistance tool**, with human review remaining part of the workflow.

---

## Future Work

Potential extensions include:

- integration with external sources for evidence-based fact verification
- Retrieval-Augmented Generation (RAG)
- speaker diarization
- timestamp-aware citations
- automatic detection of unsupported claims
- structured editorial metadata
- evaluation of transcription and generation quality
- support for additional local LLMs

---

## 📄 License

This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.
