from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
import uuid  
from PIL import Image
import pdf2image
import google.generativeai as genai
from fpdf import FPDF  
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder  # For frontend audio recording
import wave
import numpy as np

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Initialize session ID for unique responses
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generate unique session ID

def get_gemini_response(input_text, pdf_content, prompt, session_id):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([f"Session ID: {session_id}", input_text, pdf_content[0], prompt])
    return response.text

def get_gemini_response_question(prompt, session_id):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([f"Session ID: {session_id}", prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")

def generate_pdf(content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer, dest='S')
    pdf_buffer.seek(0)
    return pdf_buffer

def get_all_query(query):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([query])
    return response.text

# Streamlit UI
st.set_page_config(page_title="Your Career Helper")
st.header("MY A5 PERSONAL ATS")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

query = st.text_input("HelpDesk", key="text_query")

# Frontend Audio Recorder
st.subheader("Voice Input")
audio_data = mic_recorder(start_prompt="Click to Speak", stop_prompt="Stop Recording", key="mic")
if audio_data and isinstance(audio_data, dict) and "bytes" in audio_data:
    audio_bytes = audio_data["bytes"]  # Extract binary data
    st.success("Audio Recorded Successfully!")
    
    # Save audio to WAV file for compatibility
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)
    
    # Process audio file using speech recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio)
            st.write("**Recognized Text:**", recognized_text)
            query = recognized_text  # Set recognized text as query
    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Please try again in a quiet environment.")
    except sr.RequestError:
        st.warning("Error connecting to speech recognition service.")
    except Exception as e:
        st.warning(f"An error occurred: {e}")

# Button to submit the query
if st.button("Ask") or query:
    if query:
        response = get_all_query(query)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter or speak a query!")
