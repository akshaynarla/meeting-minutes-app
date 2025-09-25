import streamlit as st
import os, tempfile

from text_to_meetmins import make_minutes_from_text

with st.sidebar:
    st.header("Minutes")
    st.subheader("Use existing transcript for minutes:")
    out_dir = st.text_input("Output folder", value=st.session_state.get("last_out_dir", "outputs"))
    uploaded_transcript = st.file_uploader("Transcript file (.txt or .md)", type=["txt","md"])
    llm_model = st.selectbox("Ollama model", ["llama3.1:8b", "gemma3:1b"], index=0)
    llm_base  = st.text_input("Ollama base URL", value="http://localhost:11434")

st.subheader("3) Meeting Minutes")
load_transcript_btn = st.button("Load Existing Transcript")
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
    st.session_state.conversation_path = None
    st.session_state.last_out_dir = out_dir
    st.success(f"Loaded transcript: {uploaded_transcript.name}")
    st.stop() 

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