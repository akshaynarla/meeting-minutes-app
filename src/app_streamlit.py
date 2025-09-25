import os, tempfile
import streamlit as st

# LOGO = "resources/ias.jpg"
# LOGO_LINK = "https://www.ias.uni-stuttgart.de"

# streamlit run app_streamlit.py - runs the app/server locally created for making the meeting minutes using the above
# modules (which are the buttons in this app)
# set_page_config -- used for the title on the tab.
st.set_page_config(page_title="Meeting Assistant", page_icon="🗣️", layout="wide")
st.title("🗣️ Meeting Assistant")

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
    st.session_state.minutes = None
if "audio_temp_path" not in st.session_state:
    st.session_state.audio_temp_path = None  # temp file path of uploaded audio
if "last_out_dir" not in st.session_state:
    st.session_state.last_out_dir = "outputs"   # remember across pages

# https://docs.streamlit.io/get-started/fundamentals/additional-features
main_page = st.Page("app_streamlit_page1.py", title="Transcript", icon="📝")
page_2 = st.Page("app_streamlit_page2.py", title="Speaker Identification", icon="🗣️")
page_3 = st.Page("app_streamlit_page3.py", title="Meeting Minutes", icon="📋")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run the selected page
pg.run()