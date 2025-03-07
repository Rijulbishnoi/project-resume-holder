import streamlit as st
import tempfile
import os
from moviepy import editor  # Absolute import
import librosa
import numpy as np

st.title("Video Analysis App")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.video(temp_file_path)

    try:
        # Load video
        clip = editor.VideoFileClip(temp_file_path)
        audio_path = temp_file_path.replace(".mp4", ".wav")
        
        # Extract audio
        clip.audio.write_audiofile(audio_path)

        # Load audio for analysis
        audio, sr = librosa.load(audio_path, sr=None)

        # Perform basic analysis
        duration = librosa.get_duration(y=audio, sr=sr)
        st.write(f"Audio Duration: {duration:.2f} seconds")

        # Clean up temporary files
        os.remove(audio_path)
        os.remove(temp_file_path)

    except Exception as e:
        st.error(f"Error processing the video: {e}")
