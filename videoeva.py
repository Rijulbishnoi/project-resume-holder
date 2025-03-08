import streamlit as st
from pydub import AudioSegment
import librosa
import numpy as np
import tempfile
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Function to analyze video and evaluate candidate
def evaluate_candidate(job_description, video_path):
    # Extract audio from video
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio_path = video_path.replace(".mp4", ".wav")
    audio.export(audio_path, format="wav")

    # Analyze audio using librosa
    y, sr = librosa.load(audio_path, sr=None)
    f0 = librosa.yin(y, fmin=50, fmax=400)
    valid_f0 = f0[f0 > 0]

    if len(valid_f0) > 0:
        avg_pitch = np.mean(valid_f0)
        pitch_variability = np.std(valid_f0)
        confidence_score = max(0, min(100, 100 - (pitch_variability * 1.5)))
    else:
        st.error("No valid pitch detected. Please upload a video with clear speech.")
        return

    # Use Gemini API to evaluate candidate
    prompt = f"""
You are an AI-powered hiring manager. Evaluate the candidate based on the following:
1. Job Description: {job_description}
2. Candidate's Speech Analysis:
   - Average Pitch: {avg_pitch:.2f} Hz
   - Pitch Variability: {pitch_variability:.2f}
   - Confidence Score: {confidence_score:.1f}/100
3. Candidate's Project Knowledge (extracted from their audio):
   - Analyze the candidate's audio to identify any projects they mention.
   - Summarize the projects, including:
     - Project goals
     - Technologies used
     - Outcomes or achievements
   - Assess how well these projects align with the job requirements.

Provide a detailed evaluation:
1. Is the candidate qualified for the job? (Yes/No)
2. Strengths and weaknesses based on their speech and project knowledge.
3. Recommendations for improvement, including how they can better align their skills and projects with the job requirements.
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Streamlit App
st.title("Job Candidate Evaluation")

# Job Description Input
job_description = st.text_area("Enter the Job Description:", height=200)

# Video Upload
uploaded_video = st.file_uploader("Upload the candidate's video (MP4, AVI, MOV):", type=["mp4", "avi", "mov"])

if uploaded_video and job_description:
    st.success("Video uploaded successfully!")

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    # Evaluate candidate
    evaluation = evaluate_candidate(job_description, video_path)
    st.subheader("Evaluation Results:")
    st.write(evaluation)

    # Clean up temporary files
    os.remove(video_path)
    os.remove(video_path.replace(".mp4", ".wav"))