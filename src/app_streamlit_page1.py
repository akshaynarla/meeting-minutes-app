import streamlit as st
import os, tempfile

from audio_to_text import transcribe_whisper

# ---------------- Sidebar: Inputs (arguments in CLI) ----------------
# https://docs.streamlit.io/get-started/fundamentals/main-concepts
# Streamlit has built-in widgets like file_uploader, text_input, checkbox, selectbox, 
# number_input, slider, button
with st.sidebar:
    # st.logo(image=LOGO, size="large", link= LOGO_LINK)
    # creates a file upload button with an upload section.
    st.header("Inputs")
    audio_file = st.file_uploader("Audio/Video", type=["mp3","wav","m4a","mp4","mov"])
    out_dir = st.text_input("Output folder", value="outputs")

    st.header("Transcription")
    whisper_model = st.selectbox("Whisper model", ["tiny","base","small","medium","large"], index=1)
    language = st.text_input("Language code", value="")
    translate = st.checkbox("Translate to English instead of transcribe", value=False)

st.subheader("1) Transcribe")
# st.button returns True if clicked, else False.
run_transcribe = st.button("Run Transcription")

# Upon click of the button, execute:
if run_transcribe:
    if not audio_file:
        st.error("Please upload an audio/video file.")
        st.stop()

    # Persist upload to temp file so whisper can read it. Creates temp files using the lib tempfile.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    st.session_state.audio_temp_path = tmp_path
    st.session_state.last_out_dir = out_dir

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