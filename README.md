# 🎙️ Offline Meeting Minutes

A local-first meeting transcription and minutes generator using **WhisperX** + **Ollama**.

## Privacy / Local Processing

- **Audio, transcripts, and minutes stay on the device.**
- The app calls **Ollama on localhost by default**.
- Internet connectivity is allowed (e.g., for downloading model weights), but **your meeting data is not uploaded anywhere** unless you explicitly configure a remote Ollama endpoint.

To allow a non-local Ollama endpoint, set:
```bash
export OLLAMA_ALLOW_REMOTE=1
```

---

## ✅ Features

- 🎤 Transcribe audio/video files to text (WhisperX)
- 👥 Speaker identification (diarization; optional)
- 📝 AI-generated meeting minutes with action items (Ollama)
- 🎧 Live transcription from microphone (CLI)

---

## 📦 Installation

### 1) Python dependencies

```bash
pip install streamlit whisperx torch requests faster-whisper numpy scipy sounddevice faiss-cpu sentence-transformers
```

Notes:
- Live transcription requires `sounddevice` and system PortAudio.
- WhisperX uses `ffmpeg` for many formats; install ffmpeg if needed.

### 2) Ollama (local LLM)

Install Ollama and download a model:

```bash
ollama serve
ollama pull qwen3:1.7b
```

Recommended CPU models:
- `qwen3:1.7b` (good quality / CPU-friendly)
- `phi3:mini` (often faster, lower quality)

### 3) (Optional) Speaker identification (diarization)

Diarization requires a Hugging Face token and accepting the model terms:

1. Create a token with `read` access
2. Accept the model agreements:
   - `pyannote/speaker-diarization`
   - `pyannote/segmentation`

---

## 🚀 Usage

### Start the Streamlit app

```bash
cd src
streamlit run app.py
```

Open http://localhost:8501

### Live transcription (CLI)

```bash
cd src
python live_cli.py --model tiny --device auto
```

Press `Ctrl+C` to stop.

---

## ⚙️ Configuration

### Device / Precision

- Device: `auto` will use GPU if available, otherwise CPU.
- Compute type:
  - CPU: `int8` is usually fastest
  - CUDA: `float16` is usually fastest

### Troubleshooting

**Ollama connection refused**
```bash
ollama serve
```

**Linux: sounddevice / PortAudio**
```bash
sudo apt-get install libportaudio2
```

**Out of memory**
- Use smaller Whisper model (`tiny` or `base`)
- Disable diarization (speaker identification)
- Use a smaller Ollama model

---

## 📁 Project structure

```
src/
├── app.py
├── live_cli.py
└── backend/
    ├── __init__.py
    ├── device_utils.py
    ├── backend_whisper.py
    ├── backend_llm.py
    └── live_transcript.py

outputs/
temp_uploads/
```

---

## Notes on Live VAD

Live mode uses a lightweight energy-based VAD so it can run fully offline without `torch.hub` downloads.
In noisy rooms you may need to tune `--vad-threshold` (try values between 0.3 and 0.9).


## 🙏 Acknowledgments

- [Whisper](https://github.com/openai/whisper) — OpenAI's speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Optimized Whisper
- [WhisperX](https://github.com/m-bain/whisperX) — Word-level timestamps + diarization
- [Ollama](https://ollama.com) — Local LLM inference
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection