import streamlit as st
import yt_dlp
import os
import tempfile
import glob
import whisper
import warnings
from google import genai

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Video Structure Analyzer", page_icon="🎬")
st.title("🎬 Video Structure Analyzer")
st.markdown("Paste a link OR upload a file. The AI will analyze its structure.")

api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("API Key not found. Please add GEMINI_API_KEY to Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

def process_text_to_ai(raw_text):
    if len(raw_text) < 50:
        return "Error: Text too short. Check if the video has clear speech."

    prompt = f"""
    You are a professional video structure analyst. Analyze the transcript below and output a reusable "content creation template".
    Output language must match the video language.

    Format:
    1. **Hook (first 5 seconds)**
    2. **Main Structure** (Parts and core points)
    3. **Pacing and Turning Points**
    4. **Ending Call-to-Action**
    5. **Reusable Formula for Creators**

    Transcript:
    {raw_text[:15000]}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- UI Layout ---
tab1, tab2 = st.tabs(["🔗 Analyze via Link", "📁 Upload Video/Audio File"])

with tab1:
    url_input = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Analyze Link", type="primary"):
        if not url_input:
            st.warning("Please paste a link first.")
        else:
            with st.spinner("Downloading and Analyzing... Please wait."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        raw_text = ""
                        common_opts = {
                            'quiet': True,
                            'no_warnings': True,
                            'extractor_args': {'youtube': {'player_client': ['ios', 'android', 'mweb']}}
                        }
                        
                        # Try subtitles
                        ydl_opts_sub = common_opts.copy()
                        ydl_opts_sub.update({
                            'writesubtitles': True, 'writeautomaticsub': True,
                            'subtitleslangs': ['en', 'zh-Hans', 'zh'], 'subtitlesformat': 'vtt',
                            'skip_download': True, 'outtmpl': os.path.join(tmpdir, 'sub'),
                        })
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts_sub) as ydl:
                                ydl.download([url_input])
                            vtt_files = glob.glob(os.path.join(tmpdir, '*.vtt'))
                            if vtt_files:
                                with open(vtt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                                    raw_text = " ".join([line.strip() for line in f if '-->' not in line and line.strip() and not line.startswith('WEBVTT') and not line.strip().isdigit()])
                        except: pass

                        # Try Audio
                        if len(raw_text) < 50:
                            st.info("No subtitles found. Downloading audio...")
                            ydl_opts_audio = common_opts.copy()
                            ydl_opts_audio.update({
                                'format': 'm4a/bestaudio/best',
                                'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}],
                                'outtmpl': os.path.join(tmpdir, 'audio'),
                            })
                            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                                ydl.extract_info(url_input, download=True)
                                audio_file = os.path.join(tmpdir, 'audio.mp3')
                            if os.path.exists(audio_file):
                                st.info("Transcribing audio...")
                                result = whisper_model.transcribe(audio_file, fp16=False)
                                raw_text = result['text']

                        st.markdown("### Analysis Result")
                        st.write(process_text_to_ai(raw_text))
                except Exception as e:
                    st.error(f"Download Error: {str(e)} (YouTube is likely blocking this server. Please use the Upload tab instead).")

with tab2:
    uploaded_file = st.file_uploader("Upload a video or audio file", type=['mp4', 'mp3', 'wav', 'm4a', 'mov'])
    if uploaded_file is not None:
        if st.button("Analyze File", type="primary"):
            with st.spinner("Extracting audio and transcribing... (this takes a moment)"):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save uploaded file
                        file_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.info("Transcribing audio with Whisper...")
                        result = whisper_model.transcribe(file_path, fp16=False)
                        raw_text = result['text']

                        st.markdown("### Analysis Result")
                        st.write(process_text_to_ai(raw_text))
                except Exception as e:
                    st.error(f"File Error: {str(e)}")
