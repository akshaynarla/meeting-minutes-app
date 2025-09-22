import os, tempfile
import streamlit as st

# Import your modules
from audio_to_text import transcribe_whisper
from conversation_maker import (
    run_diarization, assign_speakers_to_whisper_segments,
    apply_speaker_map, write_conversation_markdown
)
from text_to_meetmins import make_minutes_from_text

st.set_page_config(page_title="Meeting Assistant", page_icon="🗣️", layout="wide")
st.title("🗣️ Meeting Assistant")

# ---------------- Sidebar: Inputs ----------------
with st.sidebar:
    st.header("Inputs")
    audio_file = st.file_uploader("Audio/Video", type=["mp3","wav","m4a","mp4","mov"])
    out_dir = st.text_input("Output folder", value="outputs")

    st.header("Transcription")
    whisper_model = st.selectbox("Whisper model (CPU)", ["tiny","base","small","medium"], index=1)
    language = st.text_input("Language code (optional, e.g., 'en')", value="")
    translate = st.checkbox("Translate to English instead of transcribe", value=False)

    st.header("Diarization (optional)")
    do_diar = st.checkbox("Enable diarization (pyannote)", value=False)
    num_speakers = st.number_input("Known number of speakers (optional)", min_value=0, value=0, step=1)
    hf_token = st.text_input("Hugging Face READ token", type="password")

    st.header("Minutes")
    summary_mode = st.selectbox("Summary mode", ["auto","abstractive","extractive"], index=0)
    abstractive_model = st.text_input("Abstractive model", value="sshleifer/distilbart-cnn-12-6")
    key_points_n = st.slider("Key points to list", 5, 20, 8)

# Session state holders
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

col1, col2 = st.columns([1,1])
with col1:
    run_transcribe = st.button("Run Transcription")

with col2:
    if st.session_state.transcript:
        st.success("Transcription already completed.")
        st.write("**Text:**", st.session_state.transcript["text"])
        st.write("**JSON:**", st.session_state.transcript["json"])
        st.write("**SRT:**",  st.session_state.transcript["srt"])
        st.write("**VTT:**",  st.session_state.transcript["vtt"])

if run_transcribe:
    if not audio_file:
        st.error("Please upload an audio/video file.")
        st.stop()

    # Persist upload to temp file so whisper can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    with st.status("Transcribing (Whisper, CPU)…", expanded=True) as s:
        st.session_state.transcript = transcribe_whisper(
            audio_path=tmp_path,
            model_size=whisper_model,
            language=(language or None),
            translate=translate,
            output_dir=out_dir
        )
        s.update(label="Transcription complete ✅", state="complete")

    # Preview transcript head
    with open(st.session_state.transcript["text"], "r", encoding="utf-8") as f:
        preview = f.read(1200)
    st.code(preview + ("..." if len(preview) == 1200 else ""), language="markdown")

    # Downloads
    st.download_button("Download transcript (.txt)",
        file_name=os.path.basename(st.session_state.transcript["text"]),
        data=open(st.session_state.transcript["text"], "rb").read())
    st.download_button("Download subtitles (.srt)",
        file_name=os.path.basename(st.session_state.transcript["srt"]),
        data=open(st.session_state.transcript["srt"], "rb").read())
    st.download_button("Download subtitles (.vtt)",
        file_name=os.path.basename(st.session_state.transcript["vtt"]),
        data=open(st.session_state.transcript["vtt"], "rb").read())

st.markdown("---")
st.subheader("2) Conversation (Speaker-labeled)")

c1, c2 = st.columns([1,1])
with c1:
    run_diar_btn = st.button("Run Diarization + Build Conversation", disabled=not do_diar)
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

        # Preview
        with open(conv_path, "r", encoding="utf-8") as f:
            head = "".join([next(f) for _ in range(30)])
        st.code(head, language="markdown")

        st.download_button("Download conversation (.md)",
            file_name=os.path.basename(conv_path),
            data=open(conv_path, "rb").read())

st.markdown("---")
st.subheader("3) Meeting Minutes")

run_minutes = st.button("Generate Minutes")
if run_minutes:
    if not st.session_state.transcript:
        st.error("Run transcription first.")
        st.stop()

    source_txt = (
        st.session_state.conversation_path  # use conversation if available
        if st.session_state.conversation_path
        else st.session_state.transcript["text"]
    )

    with st.status("Generating minutes…", expanded=True) as s:
        st.session_state.minutes = make_minutes_from_text(
            transcript_text_path=source_txt,
            whisper_json_path=st.session_state.transcript["json"],
            out_dir=out_dir,
            summary_mode=summary_mode,
            abstractive_model=abstractive_model,
            key_points_n=key_points_n
        )
        s.update(label="Minutes ready ✅", state="complete")

    # Preview + downloads
    with open(st.session_state.minutes["minutes_md"], "r", encoding="utf-8") as f:
        preview = "".join([next(f) for _ in range(40)])
    st.code(preview, language="markdown")

    st.download_button("Download minutes (.md)",
        file_name=os.path.basename(st.session_state.minutes["minutes_md"]),
        data=open(st.session_state.minutes["minutes_md"], "rb").read())
    st.download_button("Download action items (.csv)",
        file_name=os.path.basename(st.session_state.minutes["actions_csv"]),
        data=open(st.session_state.minutes["actions_csv"], "rb").read())