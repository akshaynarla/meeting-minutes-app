# Meeting Assistant

Turn raw meeting audio into clean transcript, speaker-labeled conversations, and LLM-generated minutes all locally on your machine.

* **Transcript:** CPU Whisper transcription (or skip if you already have a transcript)
* **Speaker Identification:** One-click **WhisperX** (transcribe and identify speakers)
* **Meeting Minutes:** Summarize to Markdown using a **local Ollama** model

## Features

* Runs entirely local (aimed at CPU-only; GPU would be greeeaaatttt!!)
* Upload audio/video (.mp3, .wav, .m4a, .mp4, .mov)
* Whisper transcription to .txt and .srt
* WhisperX optional: word-level timestamps + speaker labels
* Ollama LLM minutes: Markdown + plain text
* Works with existing transcripts (.txt/.md)
* 3-page Streamlit UX with download files

## App structure


?? app\_streamlit.py # main entry û navigation + global state
?? app\_streamlit\_page1.py # Transcript page (Whisper)
?? app\_streamlit\_page2.py # Speaker Identification (WhisperX)
?? app\_streamlit\_page3.py # Meeting Minutes (Ollama)
?? audio\_to\_text.py # transcribe\_whisper(...)
?? whisperx\_conv.py # transcribe\_to\_conversation(...) via WhisperX
?? text\_to\_meetmins.py # make\_minutes\_from\_text(...) via Ollama (JSON?MD)
?? resources/
 ?? ias.jpg # example logo (optional)

If you still have conversation\_maker.py (pyannote overlap approach), it is optional as WhisperX can handle diarization also alogn with transcribing in one pass.

## Quick start

### 1. Create environment & install deps

**Python 3.10+** recommended.

# CPU PyTorch first (official index)
pip install -r requirments.txt

Pull at least one local model:

ollama pull llama3.1:8b # balanced default
# or a lighter one:
# ollama pull phi3:3.8b

**WhisperX diarization (optional)** - Accept the **pyannote** diarization model terms on Hugging Face and set a read token:

## Run the app

streamlit run app\_streamlit.py

### Page 1: Transcript

1. Upload audio/video
2. Pick model (start with base on CPU)
3. **Run Transcription** to get downloads: \*.txt, \*.srt
Tip: If you already have a transcript, you can skip this and load it on **Page 3**.

### Page 2: Speaker Identification (WhisperX)

1. Ensure HF\_TOKEN is set (or paste in sidebar)
2. **Run WhisperX**: Creates conversation.md (speaker-tagged) + raw \*.json

### Page 3: Meeting Minutes (Ollama)

1. If conversation.md exists, itÆs used automatically; else falls back to transcript .txt
   (You can also **upload** a .txt/.md here and click **Load Uploaded Transcript**.)
2. Choose **Ollama model** (e.g., llama3.1:8b)
3. **Generate Minutes**
   ? Downloads: \*\_minutes.md, \*\_minutes.txt
   *(If your minutes code exports action items, you may also get \*\_actions.csv.)*

## ? CLI (optional)

Generate minutes directly from a transcript or conversation:

python text\_to\_meetmins.py outputs/<run>\_transcript/conversation.md \
 --out\_dir outputs \
 --model "llama3.1:8b" \
 --base\_url "http://localhost:11434"

## ? Outputs

For input meeting.mp3 with out\_dir=outputs:

outputs/
?? meeting\_transcript/
? ?? meeting.txt
? ?? meeting.json
? ?? meeting.srt
? ?? meeting.vtt (if enabled)
?? meeting\_whisperx/
? ?? meeting.txt
? ?? meeting.json
? ?? conversation.md
?? meeting\_minutes\_llm/
 ?? meeting\_minutes.md
 ?? meeting\_minutes.txt

## ? Performance tips (CPU)

* Set **Language code** on Page 1 to skip auto-detect.
* Prefer **Transcribe** (not **Translate**) unless you need English ? faster.
* Keep Whisper at **base** (or **small**) for a good quality/time balance.
* WhisperX adds alignment/diarization time; run it only when you need speakers.
* Later optimization: swap to **faster-whisper (CTranslate2, int8)** for 2û4Î speedups while still staying ôWhisperö.

## Privacy

Everything runs **locally**: - Whisper/WhisperX models are downloaded once and cached on your machine. - Minutes are produced by **local Ollama** models. - No audio or transcripts leave your environment.

## Roadmap

[] Progress bars for long CPU jobs
[] Optional faster-whisper backend
[] Speaker name mapping UI on Page 2

## License & credits

* **Transcription:** OpenAI Whisper ()
* **Alignment & Diarization:** WhisperX (+ pyannote)
* **Minutes:** Ollama local LLMs
* **UI:** Streamlit