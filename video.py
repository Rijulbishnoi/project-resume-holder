import streamlit as st
import tempfile
import os
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

def extract_audio(video_path):
    """Extracts audio from a video file and saves it as a WAV file."""
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", ".wav")
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        return audio_path
    except Exception as e:
        st.error(f"❌ Error extracting audio: {e}")
        return None

def analyze_audio(audio_path):
    """Extracts features from the audio file for analysis."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y))
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
        return {"duration": duration, "energy": energy, "pitch": pitch}
    except Exception as e:
        st.error(f"❌ Error analyzing audio: {e}")
        return None

def main():
    st.title("🎤 ATS Bot - Video Analysis for Interviews")
    st.write("Upload your introduction video, and the bot will analyze your strengths and weaknesses!")
    
    uploaded_file = st.file_uploader("Upload your video (MP4 format)", type=["mp4"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
        
        st.video(temp_video_path)
        
        st.write("🔍 Extracting audio...")
        audio_path = extract_audio(temp_video_path)
        
        if audio_path:
            st.write("✅ Audio extracted successfully!")
            st.write("🎙️ Analyzing voice features...")
            analysis = analyze_audio(audio_path)
            
            if analysis:
                st.write("📊 Analysis Results:")
                st.json(analysis)
            else:
                st.error("❌ Failed to analyze audio.")
        
        os.remove(temp_video_path)
        if audio_path:
            os.remove(audio_path)

