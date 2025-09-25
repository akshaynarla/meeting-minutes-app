import streamlit as st
import os, tempfile

from whisperx_convo import transcribe_to_conversation

with st.sidebar:
    st.header("Audio")
    # Allow new upload here
    audio_file = st.file_uploader("Audio/Video (optional if already uploaded on page 1)", 
                                  type=["mp3","wav","m4a","mp4","mov"], key="audio_for_whisperx")
    whisper_model = st.selectbox("Whisper model", ["tiny","base","small","medium","large"], index=1)
    out_dir = st.text_input("Output folder", value=st.session_state.get("last_out_dir", "outputs"))
    hf_token = st.text_input("Hugging Face READ token", type="password")

st.markdown("### Use WhisperX (Transcribe + Speakers in one go)")
run_whisperx = st.button("Run WhisperX")

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

    with st.status("Running WhisperX ....", expanded=True) as s:
        res = transcribe_to_conversation(
            audio_path=tmp_path,
            out_dir=out_dir,
            model_size=whisper_model,
            device="cpu",          # or "cuda" if you have a GPU
            compute_type=None,     # auto: int8 on CPU, float16 on CUDA
            hf_token=hf_token,
            timestamps=True,
        )
        
        st.session_state.transcript = {"text": res["text"], "json": res["json"], "dir": res["dir"]}
        st.session_state.conversation_path = res["conversation"]
        st.session_state.merged_segments = None  # clear any earlier state
        s.update(label="WhisperX complete ✅", state="complete")

    st.download_button("Download conversation (.md)",
        file_name=os.path.basename(st.session_state.conversation_path),
        data=open(st.session_state.conversation_path, "rb").read())