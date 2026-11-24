import streamlit as st
import yt_dlp
import google.generativeai as genai
import os
import tempfile
import glob
import whisper
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Video Structure Analyzer", page_icon="🎬")
st.title("🎬 Video Structure Analyzer")
st.markdown("Paste a video link (YouTube, Facebook, etc.). The AI will analyze its structure and generate a template.")

# --- Load API Key securely ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("API Key not found. Please add GEMINI_API_KEY to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Load Whisper Model (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

def analyze_video(url):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_text = ""
            
            # 1. Try to get subtitles first (Fast)
            ydl_opts_sub = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'zh-Hans', 'zh', 'ja', 'ko'],
                'subtitlesformat': 'vtt',
                'skip_download': True,
                'outtmpl': os.path.join(tmpdir, 'sub'),
                'quiet': True,
                'no_warnings': True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts_sub) as ydl:
                    ydl.download([url])
                vtt_files = glob.glob(os.path.join(tmpdir, '*.vtt'))
                if vtt_files:
                    with open(vtt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line in lines:
                            if '-->' not in line and line.strip() and not line.startswith('WEBVTT') and not line.strip().isdigit():
                                raw_text += line.strip() + " "
            except:
                pass 

            # 2. If no subtitles, download audio and use Whisper (Slow but works for all videos)
            if len(raw_text) < 50:
                st.info("No subtitles found. Downloading audio and transcribing with Whisper AI (this takes a moment)...")
                ydl_opts_audio = {
                    'format': 'm4a/bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
                    'outtmpl': os.path.join(tmpdir, 'audio'),
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                    ydl.extract_info(url, download=True)
                    audio_file = os.path.join(tmpdir, 'audio.mp3')
                
                if os.path.exists(audio_file):
                    result = whisper_model.transcribe(audio_file, fp16=False)
                    raw_text = result['text']
                else:
                    return "Error: Failed to extract audio."

            if len(raw_text) < 50:
                return "Error: Text too short. Check if the video has clear speech."

            # 3. Send to Gemini
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
            response = model.generate_content(prompt)
            return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# --- Streamlit UI ---
url_input = st.text_input("Video Link:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Video", type="primary"):
    if not url_input:
        st.warning("Please paste a link first.")
    else:
        with st.spinner("Analyzing... Please wait."):
            result = analyze_video(url_input)
            st.markdown("### Analysis Result")
            st.write(result)
