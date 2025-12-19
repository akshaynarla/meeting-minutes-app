# 🎙️ Offline Meeting Minutes

A fully offline meeting transcription and summarization app using Whisper + Ollama.

**Features:**
- 🎤 Transcribe audio/video files to text
- 👥 Speaker identification (diarization)
- 📝 AI-generated meeting minutes with action items
- 🎧 Live transcription from microphone (CLI)

---

## 📦 Installation

### 1. Python Dependencies

```bash
pip install streamlit faster-whisper whisperx torch requests
pip install sounddevice  # For live transcription
```

### 2. Ollama (Local LLM)

**Install Ollama:**
- **Windows:** Download from https://ollama.com/download
- **Mac:** `brew install ollama`
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

**Download a model:**
```bash
# Recommended for CPU (fast + good quality)
ollama pull qwen3:1.7b

# Alternative options
ollama pull phi3:mini      # Fastest
ollama pull qwen3:4b       # Better quality (needs more RAM/time)
```

### 3. (Optional) Speaker Identification

Requires a HuggingFace token:
1. Create account at https://huggingface.co
2. Go to Settings → Access Tokens
3. Create a token with `read` access
4. Accept the model agreements:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation

---

## 🚀 Usage

### Start Ollama

```bash
ollama serve
```

### Run the App

```bash
cd src
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📖 How to Use

### Tab 1: Transcribe Audio

1. Upload an audio/video file (mp3, wav, m4a, mp4)
2. Select Whisper model size:
   - `tiny`/`base` — Fast, good for clear audio
   - `small`/`medium` — Better accuracy
   - `large-v3` — Best quality (slow on CPU)
3. (Optional) Enable speaker identification
4. Click **Start Processing**

### Tab 2: Generate Minutes

1. Process audio in Tab 1, OR upload a transcript file
2. Select your Ollama model (recommend `qwen3:1.7b` for CPU)
3. Click **Generate Minutes**
4. Download the markdown file

### Tab 3: Live Transcription

The Streamlit UI shows initialization only. For actual live transcription, use the CLI:

```bash
cd src
python live_cli.py
```

Speak into your microphone. Press `Ctrl+C` to stop.

---

## ⚙️ Configuration

### Whisper Models

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | ⚡⚡⚡⚡ | ★★ | ~1GB |
| base | ⚡⚡⚡ | ★★★ | ~1GB |
| small | ⚡⚡ | ★★★★ | ~2GB |
| medium | ⚡ | ★★★★ | ~5GB |
| large-v3 | 🐢 | ★★★★★ | ~10GB |

### Ollama Models

| Model | Speed (CPU) | Quality | RAM |
|-------|------------|---------|-----|
| phi3:mini | ⚡⚡⚡⚡ | ★★★ | ~2GB |
| qwen3:1.7b | ⚡⚡⚡ | ★★★★ | ~3GB |
| qwen3:4b | ⚡⚡ | ★★★★ | ~6GB |
| llama3.1:8b | ⚡ | ★★★★★ | ~10GB |

**Recommendation:** Use `qwen3:1.7b` for CPU inference.

---

## 📁 Project Structure

```
src/
├── app.py                 # Streamlit app
├── live_cli.py            # CLI live transcription
└── backend/
    ├── __init__.py
    ├── backend_whisper.py # Audio transcription
    ├── backend_llm.py     # Meeting minutes generation
    └── live_transcript.py # Live transcription backend

outputs/                   # Generated files saved here
temp_uploads/              # Temporary upload directory
```

---

## 🔧 Troubleshooting

### "Connection refused" to Ollama
```bash
# Make sure Ollama is running
ollama serve
```

### Meeting minutes taking too long
- Switch to a smaller model: `qwen3:1.7b` or `phi3:mini`
- Check if streaming is enabled in `backend_llm.py` (`"stream": True`)

### "No module named sounddevice"
```bash
pip install sounddevice

# Linux may also need:
sudo apt-get install libportaudio2
```

### Out of memory
- Use smaller Whisper model (`tiny` or `base`)
- Use smaller Ollama model (`qwen3:1.7b`)
- Close other applications

### Poor transcription quality
- Use larger Whisper model (`small` or `medium`)
- Ensure audio is clear with minimal background noise
- Check audio sample rate (16kHz is optimal)

---

## 🛠️ Technical Notes

### Why Streaming Matters

The LLM backend uses streaming (`"stream": True`) to avoid UI freezing:

```python
# Without streaming: UI freezes for 60+ seconds
resp = requests.post(url, json={"stream": False})

# With streaming: See progress in real-time
with requests.post(url, json={"stream": True}, stream=True) as resp:
    for line in resp.iter_lines():
        # Process tokens as they arrive
```

### API Endpoints

Ollama runs at `http://localhost:11434`:
- `POST /api/chat` — Chat completion
- `GET /api/tags` — List models
- `POST /api/generate` — Text generation

---

## 📄 Output Format

Generated meeting minutes include:

```markdown
# Meeting Minutes

**Date:** 2025-01-15

## 📝 Executive Summary
Brief overview of the meeting...

## 🔑 Key Points
- Point 1
- Point 2

## 🤝 Decisions Made
- Decision 1
- Decision 2

## ✅ Action Items
| Task | Owner | Due |
|------|-------|-----|
| Complete report | Alice | Friday |
```

---

## 📜 License

MIT License - Use freely for personal and commercial projects.

---

## 🙏 Acknowledgments

- [Whisper](https://github.com/openai/whisper) — OpenAI's speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Optimized Whisper
- [WhisperX](https://github.com/m-bain/whisperX) — Word-level timestamps + diarization
- [Ollama](https://ollama.com) — Local LLM inference
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection