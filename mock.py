import streamlit as st
import os
import io
import time
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
import csv
from datetime import datetime
import threading

# Thread-safe CSV writer lock
csv_lock = threading.Lock()

# Initialize logging file
LOG_FILE = "api_usage_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Action', 'API_Hits', 'Tokens_Generated', 'Time_Taken(seconds)'])

def log_api_usage(action, api_hits, tokens_generated, time_taken):
    """Log API usage details to CSV file."""
    with csv_lock:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), action, api_hits, tokens_generated, f"{time_taken:.2f}"])

# Load environment variables
load_dotenv()

# Configure APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    st.warning("ELEVENLABS_API_KEY not found. Text-to-speech will be disabled, but interview will proceed without audio.")

# Default voice ID (fallback)
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Common default voice in ElevenLabs free tier

def get_available_voices():
    """Fetch available voices from ElevenLabs API."""
    if not ELEVENLABS_API_KEY:
        return [("Default", DEFAULT_VOICE_ID)]
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            voices = response.json().get("voices", [])
            voice_list = [(voice["name"], voice["voice_id"]) for voice in voices]
            if not voice_list:
                return [("Default", DEFAULT_VOICE_ID)]
            return voice_list
        else:
            st.warning(f"Error fetching voices: HTTP {response.status_code} - {response.text}")
            return [("Default", DEFAULT_VOICE_ID)]
    except Exception as e:
        st.warning(f"Error fetching voices: {str(e)}")
        return [("Default", DEFAULT_VOICE_ID)]

# Text-to-Speech Function (HTTP-based ElevenLabs API)
def text_to_speech(text, voice_id=DEFAULT_VOICE_ID):
    """Convert text to speech using ElevenLabs API with HTTP requests and logging."""
    if not ELEVENLABS_API_KEY:
        return None, "⚠ ElevenLabs API key not configured."

    start_time = time.time()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            audio_bytes = io.BytesIO(response.content)
            audio_bytes.seek(0)
            tokens = len(text) // 4
            log_api_usage("ElevenLabs_TTS_HTTP", 1, tokens, time.time() - start_time)
            return audio_bytes, None
        else:
            error_msg = f"Error: HTTP {response.status_code} - {response.text}"
            log_api_usage("ElevenLabs_TTS_HTTP_Error", 1, 0, time.time() - start_time)
            # Fallback to default voice if voice_id is invalid
            if "voice_not_found" in error_msg and voice_id != DEFAULT_VOICE_ID:
                return text_to_speech(text, voice_id=DEFAULT_VOICE_ID)
            return None, error_msg
    except Exception as e:
        log_api_usage("ElevenLabs_TTS_HTTP_Error", 1, 0, time.time() - start_time)
        return None, f"Error generating audio: {str(e)}"

# Gemini Response Function
def get_gemini_response(prompt):
    """Generate a response using Google Gemini API with logging."""
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    start_time = time.time()
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add unique variations each time this prompt is called: {os.urandom(8).hex()}"])
        tokens_generated = len(response.text) // 4 if hasattr(response, 'text') and response.text else 0
        log_api_usage("Gemini_API_Call", 1, tokens_generated, time.time() - start_time)
        if hasattr(response, 'text') and response.text:
            return response.text
        return "Error: No valid response received from Gemini API."
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        log_api_usage("Gemini_API_Error", 1, 0, time.time() - start_time)
        return f"Error: {str(e)}"

# Process Audio Input
def process_audio(audio_dict, text_key, question_index):
    """Process audio input and return recognized text."""
    if audio_dict and "bytes" in audio_dict:
        try:
            audio_bytes = audio_dict["bytes"]
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source)
                recognized_text = recognizer.recognize_google(audio)
                st.session_state[text_key] = recognized_text
                st.text_area(f"Recognized Answer for Question {question_index + 1}:", recognized_text, key=f"{text_key}_area_{question_index}")
                return recognized_text
        except sr.UnknownValueError:
            st.warning("Could not understand the audio. Please try again in a quiet environment.")
        except sr.RequestError:
            st.warning("Error connecting to the speech recognition service.")
        except Exception as e:
            st.warning(f"An error occurred: {e}")
    return None

# Generate Question
def generate_question(level, topic, question_num):
    """Generate a unique theoretical interview question with logging."""
    start_time = time.time()
    query = f"Generate a concise, one-line {level} level theoretical interview question (Question {question_num}/3) for a distinct subtopic of {topic} that assesses the candidate's conceptual understanding. Ensure the question is unique, focuses on a different aspect or subtopic than any previous questions in this set, and has not been asked before in this session. For Digital Marketing, include questions about API usage, automation, or analytics where relevant. Previously asked questions: {', '.join(st.session_state.used_questions) if st.session_state.used_questions else 'None'}."
    question = get_gemini_response(query)
    tokens = len(question) // 4
    log_api_usage(f"Generate_Question_{question_num}", 1, tokens, time.time() - start_time)
    st.session_state.used_questions.append(question)
    return question

# Mock Interview System
st.title("AI-Powered Automated Mock Interview System")

# Initialize session states
if 'recognized_text_2' not in st.session_state:
    st.session_state.recognized_text_2 = ""
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'started' not in st.session_state:
    st.session_state.started = False
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = "Easy"
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False
if 'audio_dict_mock' not in st.session_state:
    st.session_state.audio_dict_mock = None
if 'used_questions' not in st.session_state:
    st.session_state.used_questions = []
if 'waiting_for_answer' not in st.session_state:
    st.session_state.waiting_for_answer = False

# User inputs
topic = st.radio("Select Topic:", ("Python", "SQL", "Digital Marketing"), key="mock_topic")
level = st.radio("Select Difficulty:", ("Easy", "Intermediate", "Hard"), key="mock_level", index=["Easy", "Intermediate", "Hard"].index(st.session_state.difficulty))

# Fetch available voices
voice_options = get_available_voices()
voice_names = [name for name, _ in voice_options]
voice_id_map = {name: voice_id for name, voice_id in voice_options}
selected_voice_name = st.selectbox("Select Interviewer Voice:", voice_names, key="voice_selection")
selected_voice_id = voice_id_map.get(selected_voice_name, DEFAULT_VOICE_ID)

# Start button
if st.button("Start Interview", key="start_interview"):
    with st.spinner("Generating questions..."):
        st.session_state.questions = [
            generate_question(level, topic, 1),
            generate_question(level, topic, 2),
            generate_question(level, topic, 3)
        ]
        st.session_state.answers = []
        st.session_state.current_question_index = 0
        st.session_state.started = True
        st.session_state.interview_complete = False
        st.session_state.audio_dict_mock = None
        st.session_state.waiting_for_answer = True
        st.rerun()

# Automated Interview Logic
if st.session_state.started and not st.session_state.interview_complete:
    if st.session_state.current_question_index < 3:
        current_question = st.session_state.questions[st.session_state.current_question_index]
        
        # Play question audio (no text display)
        audio_bytes, error = text_to_speech(current_question, voice_id=selected_voice_id)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        elif error:
            st.warning(error)
            # Fallback: display question text if audio fails
            st.write(f"**Question {st.session_state.current_question_index + 1}/3:** {current_question}")

        # Record user's answer
        if st.session_state.waiting_for_answer:
            st.session_state.audio_dict_mock = mic_recorder(
                start_prompt=f"Click to Speak Your Answer for Question {st.session_state.current_question_index + 1}",
                stop_prompt="Stop Recording",
                key=f"mic_mock_interview_{st.session_state.current_question_index}"
            )

            if st.session_state.audio_dict_mock:
                with st.spinner("Processing your answer..."):
                    recognized_text_mock = process_audio(st.session_state.audio_dict_mock, "recognized_text_2", st.session_state.current_question_index)
                    if recognized_text_mock:
                        st.session_state.answers.append(recognized_text_mock)
                        start_time = time.time()
                        evaluation = get_gemini_response(f"Evaluate this answer in terms of correctness, clarity, and depth for the question '{current_question}': {recognized_text_mock}")
                        tokens = len(evaluation) // 4
                        log_api_usage(f"Evaluate_Answer_Q{st.session_state.current_question_index + 1}", 1, tokens, time.time() - start_time)
                        st.subheader(f"Evaluation for Question {st.session_state.current_question_index + 1}:")
                        st.write(evaluation)

                        # Adjust difficulty based on evaluation
                        if "good" in evaluation.lower() and st.session_state.difficulty != "Hard":
                            st.session_state.difficulty = "Intermediate" if st.session_state.difficulty == "Easy" else "Hard"
                        elif "poor" in evaluation.lower() and st.session_state.difficulty != "Easy":
                            st.session_state.difficulty = "Easy" if st.session_state.difficulty == "Hard" else "Intermediate"

                        # Move to next question
                        st.session_state.current_question_index += 1
                        st.session_state.audio_dict_mock = None
                        st.session_state.waiting_for_answer = True
                        if st.session_state.current_question_index < 3:
                            st.rerun()
                        else:
                            st.session_state.interview_complete = True
                            st.rerun()

# Interview Completion
if st.session_state.interview_complete and len(st.session_state.answers) == 3:
    st.subheader("Interview Completed!")
    combined_answers = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers))])
    start_time = time.time()
    feedback = get_gemini_response(f"Provide overall feedback for these 3 question-answer pairs and suggest improvements:\n{combined_answers}")
    tokens = len(feedback) // 4
    log_api_usage("Overall_Feedback", 1, tokens, time.time() - start_time)
    st.subheader("Overall Feedback:")
    st.write(feedback)

    # Play feedback audio
    concluding_message = "Thank you for completing the mock interview. Below is your overall feedback."
    audio_bytes, error = text_to_speech(concluding_message, voice_id=selected_voice_id)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    elif error:
        st.warning(error)

    if st.button("Restart Interview", key="restart_interview"):
        st.session_state.started = False
        st.session_state.interview_complete = False
        st.session_state.current_question_index = 0
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.audio_dict_mock = None
        st.session_state.used_questions = []
        st.session_state.waiting_for_answer = False
        st.rerun()