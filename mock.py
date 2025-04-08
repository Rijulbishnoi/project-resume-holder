import streamlit as st
import io
from pydub import AudioSegment
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Session state initialization
if 'recognized_text_1' not in st.session_state:
    st.session_state.recognized_text_1 = ""  # For general voice queries
if 'recognized_text_2' not in st.session_state:
    st.session_state.recognized_text_2 = ""  # For mock interview answers
if 'answers' not in st.session_state:
    st.session_state.answers = []  # Store answers for all questions
if 'questions' not in st.session_state:
    st.session_state.questions = []  # Store the three questions
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0  # Track which question is being answered
if 'started' not in st.session_state:
    st.session_state.started = False
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = "Easy"
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False
if 'audio_dict_mock' not in st.session_state:
    st.session_state.audio_dict_mock = None  # Store audio dict for mock interview

# Function to get response from Gemini API
def get_gemini_response(prompt):
    """Generate a response using Google Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Ensure uniqueness with random seed: {os.urandom(4).hex()}"])
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return f"Error: {str(e)}"

# Audio processing function
def process_audio(audio_dict, text_key, question_index):
    """Process audio input and return recognized text."""
    if audio_dict and "bytes" in audio_dict:
        st.success("Audio Recorded Successfully!")
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

# Function to generate a unique interview question
def generate_question(level, topic, question_num):
    """Generate a unique theoretical interview question."""
    query = f"Ask a concise, one-line {level} level theoretical {topic} interview question (Question {question_num}/3) that assesses the candidate’s conceptual understanding. Ensure it’s unique and different from previous questions."
    return get_gemini_response(query)

# General Voice Query Section
st.subheader("Voice Query")
audio_dict = mic_recorder(start_prompt="Click to Speak", stop_prompt="Stop Recording", key="mic_general")
recognized_text = process_audio(audio_dict, "recognized_text_1", 0)  # Using index 0 for general query
if recognized_text:
    response = get_gemini_response(recognized_text)
    st.subheader("Response:")
    st.write(response)

# Text input fallback
query = st.text_input("HelpDesk", key="text_query")
if st.button("Ask", key="ask_query"):
    if query:
        response = get_gemini_response(query)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter or speak a query!")

# Mock Interview System
st.title("AI-Powered Mock Interview System")

# User selects topic & difficulty level with unique keys
topic = st.radio("Select Topic:", ("Python", "SQL"), key="mock_topic")
level = st.radio("Select Difficulty:", ("Easy", "Intermediate", "Hard"), key="mock_level", index=["Easy", "Intermediate", "Hard"].index(st.session_state.difficulty))

# Start the interview and generate 3 unique questions
if st.button("Start Interview", key="start_interview"):
    st.session_state.questions = [
        generate_question(st.session_state.difficulty, topic, 1),
        generate_question(st.session_state.difficulty, topic, 2),
        generate_question(st.session_state.difficulty, topic, 3)
    ]
    st.session_state.answers = []  # Reset answers
    st.session_state.current_question_index = 0
    st.session_state.started = True
    st.session_state.interview_complete = False
    st.session_state.audio_dict_mock = None  # Reset audio dict
    st.rerun()

# Mock Interview Section
if st.session_state.started and not st.session_state.interview_complete:
    if st.session_state.current_question_index < 3:
        current_question = st.session_state.questions[st.session_state.current_question_index]
        st.write(f"**Question {st.session_state.current_question_index + 1}/3:** {current_question}")

        # Single button to trigger recording directly
        st.session_state.audio_dict_mock = mic_recorder(
            start_prompt=f"Click to Speak Your Answer for Question {st.session_state.current_question_index + 1}",
            stop_prompt="Stop Recording",
            key=f"mic_mock_interview_{st.session_state.current_question_index}"
        )

        # Process the recorded audio if available
        if st.session_state.audio_dict_mock:
            recognized_text_mock = process_audio(st.session_state.audio_dict_mock, "recognized_text_2", st.session_state.current_question_index)
            if recognized_text_mock:
                st.session_state.answers.append(recognized_text_mock)  # Store answer
                evaluation = get_gemini_response(f"Evaluate this answer in terms of correctness, clarity, and depth for the question '{current_question}': {recognized_text_mock}")
                st.subheader(f"Evaluation for Question {st.session_state.current_question_index + 1}:")
                st.write(evaluation)

                # Adjust difficulty for next question
                if "good" in evaluation.lower() and st.session_state.difficulty != "Hard":
                    st.session_state.difficulty = "Intermediate" if st.session_state.difficulty == "Easy" else "Hard"
                elif "poor" in evaluation.lower() and st.session_state.difficulty != "Easy":
                    st.session_state.difficulty = "Easy" if st.session_state.difficulty == "Hard" else "Intermediate"

                # Move to next question and reset audio dict
                st.session_state.current_question_index += 1
                st.session_state.audio_dict_mock = None  # Clear audio dict to allow new recording
                if st.session_state.current_question_index < 3:
                    st.rerun()
                else:
                    st.session_state.interview_complete = True
                    st.rerun()

# Provide overall feedback after all 3 questions are answered
if st.session_state.interview_complete and len(st.session_state.answers) == 3:
    st.subheader("Interview Completed!")
    combined_answers = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers))])
    feedback = get_gemini_response(f"Provide overall feedback for these 3 question-answer pairs and suggest improvements:\n{combined_answers}")
    st.subheader("Overall Feedback:")
    st.write(feedback)

    # Option to restart
    if st.button("Restart Interview", key="restart_interview"):
        st.session_state.started = False
        st.session_state.interview_complete = False
        st.session_state.current_question_index = 0
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.audio_dict_mock = None
        st.rerun()