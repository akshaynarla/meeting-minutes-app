import os, tempfile
import streamlit as st

# Import modules necessary for the app
from audio_to_text import transcribe_whisper
from conversation_maker import (
    run_diarization, assign_speakers_to_whisper_segments,
    apply_speaker_map, write_conversation_markdown
)
from text_to_meetmins import make_minutes_from_text
from whisperx_convo import transcribe_to_conversation

LOGO = "resources/ias.jpg"
LOGO_LINK = "https://www.ias.uni-stuttgart.de"

# streamlit run app_streamlit.py - runs the app/server locally created for making the meeting minutes using the above
# modules (which are the buttons in this app)
# set_page_config -- used for the title on the tab.
st.set_page_config(page_title="Meeting Assistant", page_icon=LOGO, layout="wide")
st.title("🗣️ Meeting Assistant")

# ---------------- Sidebar: Inputs (arguments in CLI) ----------------
# https://docs.streamlit.io/get-started/fundamentals/main-concepts
# Streamlit has built-in widgets like file_uploader, text_input, checkbox, selectbox, number_input, slider, button
with st.sidebar:
    st.logo(image=LOGO, size="large", link= LOGO_LINK)
    
    st.header("Inputs")
    audio_file = st.file_uploader("Audio/Video", type=["mp3","wav","m4a","mp4","mov"])        # creates a file upload button with an upload section.
    out_dir = st.text_input("Output folder", value="outputs")

    st.header("Transcription")
    whisper_model = st.selectbox("Whisper model", ["tiny","base","small","medium","large"], index=1)
    language = st.text_input("Language code", value="")
    translate = st.checkbox("Translate to English instead of transcribe", value=False)

    st.header("Speaker Identification")
    do_diar = st.checkbox("Enable Speaker Identification", value=False)
    num_speakers = st.number_input("Known number of speakers", min_value=0, value=0, step=1)
    hf_token = st.text_input("Hugging Face READ token", type="password")

    st.header("Minutes")
    st.subheader("Use existing transcript for minutes:")
    uploaded_transcript = st.file_uploader("Transcript file (.txt or .md)", type=["txt","md"])
    llm_model = st.text_input("Ollama model", value="llama3.1:8b")
    llm_base  = st.text_input("Ollama base URL", value="http://localhost:11434")

# Session state holders: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts#session-state
# holds the tab details. Opening a new session will create a new session.
if "transcript" not in st.session_state:
    st.session_state.transcript = None  # dict of paths from transcribe_whisper
if "merged_segments" not in st.session_state:
    st.session_state.merged_segments = None  # [{start,end,speaker,text}, ...]
if "speaker_mapping" not in st.session_state:
    st.session_state.speaker_mapping = {}    # {"SPEAKER_00":"Akshay",...}
if "conversation_path" not in st.session_state:
    st.session_state.conversation_path = None
if "minutes" not in st.session_state:
    st.session_state.minutes = None          # dict of paths from make_minutes_from_text

# ---------------- Main UI ----------------
st.subheader("1) Transcribe")
# col1 and col2: would be side-by-side. Here button on left and the output on right.
# st.button returns True if clicked, else False.
col1, col2 = st.columns([1,1])
with col1:
    run_transcribe = st.button("Run Transcription")

with col2:
    load_transcript_btn = st.button("Load Uploaded Transcript")

if load_transcript_btn:
    if not uploaded_transcript:
        st.error("Upload a .txt or .md transcript in the sidebar first.")
        st.stop()

    suffix = os.path.splitext(uploaded_transcript.name)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_transcript.getbuffer())
        tmp_path = tmp.name

    # Mimic the structure for further steps
    st.session_state.transcript = {
        "text": tmp_path,
        "json": "",
        "dir": os.path.dirname(tmp_path)
    }
    st.session_state.conversation_path = None  # reset if user switches sources

    st.success(f"Loaded transcript: {uploaded_transcript.name}")
    st.stop() 

# Upon click of the button, execute:
if run_transcribe:
    if not audio_file:
        st.error("Please upload an audio/video file.")
        st.stop()

    # Persist upload to temp file so whisper can read it. Creates temp files using the lib tempfile.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    with st.status("Transcribing using Whisper…", expanded=True) as s:
        st.session_state.transcript = transcribe_whisper(
            audio_path=tmp_path,
            model_size=whisper_model,
            language=(language or None),
            translate=translate,
            output_dir=out_dir
        )
        s.update(label="Transcription complete ✅", state="complete")

    # Downloads
    st.download_button("Download transcript (.txt)",
        file_name=os.path.basename(st.session_state.transcript["text"]),
        data=open(st.session_state.transcript["text"], "rb").read())
    st.download_button("Download subtitles (.srt)",
        file_name=os.path.basename(st.session_state.transcript["srt"]),
        data=open(st.session_state.transcript["srt"], "rb").read())
    
st.markdown("---")

st.markdown("### 1b) Or use WhisperX (Transcribe + Speakers in one go)")
run_whisperx = st.button("Run WhisperX (ASR + Diarization)")

if run_whisperx:
    if not audio_file:
        st.error("Please upload an audio/video file.")
        st.stop()
    if not hf_token:
        st.error("Hugging Face READ token required for WhisperX diarization.")
        st.stop()

    # save upload to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    with st.status("Running WhisperX (ASR + alignment + diarization)…", expanded=True) as s:
        res = transcribe_to_conversation(
            audio_path=tmp_path,
            out_dir=out_dir,
            model_size="base",     # good default for CPU; change to "small"/"medium" if needed
            device="cpu",          # or "cuda" if you have a GPU
            compute_type=None,     # auto: int8 on CPU, float16 on CUDA
            hf_token=hf_token,
            timestamps=True,
        )
        # Fill session like the rest of your app expects
        st.session_state.transcript = {"text": res["text"], "json": res["json"], "dir": res["dir"]}
        st.session_state.conversation_path = res["conversation"]
        st.session_state.merged_segments = None  # clear any earlier state
        s.update(label="WhisperX complete ✅", state="complete")

    st.download_button("Download conversation (.md)",
        file_name=os.path.basename(st.session_state.conversation_path),
        data=open(st.session_state.conversation_path, "rb").read())

st.subheader("2) Conversation (Speaker-labeled)")

c1, c2 = st.columns([1,1])
with c1:
    run_diar_btn = st.button("Run Speaker Identification + Build Conversation", disabled=not do_diar)
with c2:
    if not do_diar:
        st.info("Turn on **Enable diarization** in the sidebar to use this step.")

if run_diar_btn:
    if not st.session_state.transcript:
        st.error("Run transcription first.")
        st.stop()
    if not hf_token:
        st.error("Hugging Face READ token required for pyannote diarization.")
        st.stop()

    # Reuse same temp file (streamlit upload) by writing again
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    with st.status("Diarizing & mapping speakers…", expanded=True) as s:
        turns = run_diarization(
            audio_path=tmp_path,
            hf_token=hf_token,
            num_speakers=(num_speakers or None)
        )
        merged = assign_speakers_to_whisper_segments(
            st.session_state.transcript["json"], turns
        )
        st.session_state.merged_segments = merged

        # Build a default mapping (empty values -> will show in UI)
        unique_speakers = sorted({m["speaker"] for m in merged})
        st.session_state.speaker_mapping = {spk: st.session_state.speaker_mapping.get(spk, "") for spk in unique_speakers}
        s.update(label="Diarization complete ✅", state="complete")

# Speaker rename UI
if st.session_state.merged_segments:
    st.write("**Rename speakers** (leave blank to keep original labels):")
    cols = st.columns(min(4, len(st.session_state.speaker_mapping) or 1))
    i = 0
    new_map = {}
    for spk, current in st.session_state.speaker_mapping.items():
        with cols[i % len(cols)]:
            new_map[spk] = st.text_input(f"{spk}", value=current)
        i += 1

    # Apply mapping and write conversation
    if st.button("Apply Names & Save Conversation"):
        mapped = apply_speaker_map(st.session_state.merged_segments, new_map)
        # Save under the transcript run dir
        conv_path = os.path.join(st.session_state.transcript["dir"], "conversation.md")
        write_conversation_markdown(mapped, conv_path, timestamps=True)
        st.session_state.conversation_path = conv_path
        st.session_state.speaker_mapping = new_map
        st.success(f"Saved conversation → {conv_path}")
        st.download_button("Download conversation (.md)",
            file_name=os.path.basename(conv_path),
            data=open(conv_path, "rb").read())

st.markdown("---")
st.subheader("3) Meeting Minutes")

run_minutes = st.button("Generate Minutes")
if run_minutes:
    source_txt = (
        st.session_state.conversation_path
        if st.session_state.conversation_path
        else (st.session_state.transcript["text"] if st.session_state.transcript else None)
    )

    if not source_txt:
        st.error("No transcript available. Transcribe audio or load a .txt/.md transcript first.")
        st.stop()

    with st.status("Calling Ollama and assembling minutes…", expanded=True) as s:
        res = make_minutes_from_text(
            transcript_text_path=source_txt,
            out_dir=out_dir,
            model=llm_model,
            base_url=llm_base,
            title="Meeting Minutes",
        )
        s.update(label="Minutes ready ✅", state="complete")

    st.download_button("Download Minutes (.md)",
        file_name=os.path.basename(res["minutes_md"]),
        data=open(res["minutes_md"], "rb").read())
    st.download_button("Download Minutes (.txt)",
        file_name=os.path.basename(res["minutes_txt"]),
        data=open(res["minutes_txt"], "rb").read())