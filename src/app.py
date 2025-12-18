# src/app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from backend import process_audio, generate_minutes, LiveTranscriber

st.set_page_config(page_title="Meeting Assistant", page_icon="🎙️", layout="wide")

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("### Audio Model")
    model_size = st.selectbox("Whisper Model", ["base", "small", "medium", "large-v3", "distil-large-v3"], index=0)
    compute_type = st.selectbox("Precision (CPU)", ["int8", "float32"], index=0)

    st.markdown("### Diarization (Speaker ID)")
    enable_diarization = st.checkbox("Identify Speakers", value=False)
    hf_token = st.text_input("Hugging Face Token", type="password", help="Required for Speaker ID")

    st.markdown("### LLM (Ollama)")
    ollama_model = st.selectbox("Model Name", ["qwen3:4b", "llama3.1:8b"], index=0)
    ollama_url = st.text_input("Base URL", value="http://localhost:11434")

tab1, tab2, tab3 = st.tabs(["🎙️ Process Audio", "📝 Generate Minutes", "🎧 Live"])

# === TAB 1: Process Audio ===
with tab1:
    st.header("Upload & Transcribe")
    uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "m4a", "mp4"])

    if st.button("Start Processing", type="primary"):
        if not uploaded_file:
            st.error("Please upload a file first.")
        elif enable_diarization and not hf_token:
            st.error("You checked 'Identify Speakers' but didn't provide a Hugging Face token.")
        else:
            temp_dir = "temp_uploads"; os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

            with st.status("Processing…", expanded=True) as status:
                try:
                    results = process_audio(
                        audio_path=temp_path,
                        model_size=model_size,
                        device="cpu",
                        compute_type=compute_type,
                        diarize=enable_diarization,
                        hf_token=hf_token
                    )
                    st.session_state.processed_data = results
                    status.update(label="Complete!", state="complete")
                    st.success("Processing finished.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    if st.session_state.processed_data:
        data = st.session_state.processed_data
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Conversation")
            with open(data["conversation"], "r", encoding="utf-8") as f:
                st.text_area("Speaker-labeled transcript", f.read(), height=300)
        with col2:
            st.subheader("Downloads")
            st.download_button("Download Text", open(data["text"], "rb"), "transcript.txt")
            st.download_button("Download JSON", open(data["json"], "rb"), "data.json")
            st.download_button("Download Conversation (MD)", open(data["conversation"], "rb"), "conversation.md")

# === TAB 2: Generate Minutes ===
with tab2:
    st.header("AI Meeting Minutes")
    transcript_source = None
    if st.session_state.processed_data:
        transcript_source = st.session_state.processed_data["conversation"]
        st.info(f"Using: {os.path.basename(transcript_source)}")
    else:
        upl = st.file_uploader("Or upload a transcript (.txt/.md)", type=["txt", "md"])
        if upl:
            os.makedirs("temp_uploads", exist_ok=True)
            t_path = os.path.join("temp_uploads", "uploaded_transcript.txt")
            with open(t_path, "wb") as f:
                f.write(upl.getbuffer())
            transcript_source = t_path

    if st.button("Generate Minutes"):
        if not transcript_source:
            st.warning("No transcript available. Process audio in Tab 1 or upload a file.")
        else:
            try:
                minutes_path = generate_minutes(transcript_source, model=ollama_model, base_url=ollama_url)
                with open(minutes_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                st.markdown(md_content)
                st.download_button("Download Minutes", md_content, "minutes.md")
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")

# === TAB 3: Live (placeholder) ===
with tab3:
    st.header("Live Transcription")
    st.write("This initializes the live backend; wiring mic streaming can be added later.")
    if st.button("Start Live Session (Init Only)"):
        try:
            live = LiveTranscriber(model_size="tiny", device="cpu")
            st.success("Live backend initialized.")
        except Exception as e:
            st.error(f"Failed to init live backend: {e}")
