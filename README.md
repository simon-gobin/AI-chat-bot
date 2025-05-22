# AI Chat Bot (Local LLM + GPU)

This project is a locally hosted chatbot that runs fully offline, powered by GPU acceleration. It combines structured prompting, local LLMs, and image generation in a modular Python pipeline.

The goal is to build a lightweight assistant that can handle conversation, image generation, and summarization — with no need for external APIs.

---

## Features

- Local execution, no cloud calls
- GPU support (tested on RTX 3060 Ti)
- Two-stage LLM pipeline:
  - Qwen3 for structure and logic
  - Nous Hermes 2 Mistral for replies
- Image generation using a Stable Diffusion-based model (Flux.1-dev)
- JSON-based message history
- Automatic conversation summarization with BART
- Planned integration with a Streamlit UI

---

## Tech Stack

- Python 3.10+
- `transformers`, `torch`, `sentencepiece`, `diffusers`
- `llama-cpp-python` (for quantized local models)
- `BART` for summarization
- Streamlit or FastAPI planned for frontend

---

## Project Structure

```
AI-chat-bot/
├── chat_history/        # JSON logs
├── models/              # LLMs (quantized)
├── image_gen/           # Image generation tools
├── summarizer/          # Summarization scripts
├── ui/                  # UI layer (optional)
├── main.py              # Core loop
├── config.json          # Prompt structure
└── requirements.txt
```

---

## Setup

Clone the repo:

```bash
git clone https://github.com/simon-gobin/AI-chat-bot.git
cd AI-chat-bot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure your GPU and CUDA drivers are properly installed.

---

## Run the Bot

```bash
python main.py
```

> The system loads the models, handles conversation, and stores chat logs locally.

---

## Models Used

- **Qwen3-8B** – for context and flow control
- **Nous Hermes 2 Mistral** – for response generation
- **Stable Diffusion (Flux.1-dev)** – for image generation
- **BART (facebook/bart-large-cnn)** – for summarizing longer dialogues

---

## Roadmap

- Add Streamlit UI (chat format)
- Optional API support via FastAPI
- LoRA training support
- Docker container for deployment

---

## About

Built by **Simon Gobin**, fraud specialist and AI student, with a focus on local, secure AI applications.

- LinkedIn: [simon-gobin](https://www.linkedin.com/in/simongobin)
- Portfolio: [simongobin.wordpress.com](https://simongobin.wordpress.com)

---

## License

MIT License
