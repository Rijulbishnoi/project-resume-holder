import os
from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Initialize session state variables
if 'recognized_texts' not in st.session_state:
    st.session_state.recognized_texts = {}  # To store recognized text for each question

if 'questions' not in st.session_state:
    st.session_state.questions = []  # To store the list of questions

if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 3  # Default number of questions

if 'submitted' not in st.session_state:
    st.session_state.submitted = False  # To track if the user has submitted answers

if 'topic' not in st.session_state:
    st.session_state.topic = "Python"  # Default topic

if 'difficulty' not in st.session_state:
    st.session_state.difficulty = "Easy"  # Default difficulty

if 'recording' not in st.session_state:
    st.session_state.recording = False  # To track recording state

# Function to generate unique questions using Gemini API
def generate_question(topic, difficulty, question_number):
    """Generate a unique question using the Gemini API."""
    try:
        prompt = f"""
        Generate question which should be different from one another  {difficulty.lower()} level interview question on {topic}. 
        The question should be theoretical in nature and cover a different aspect of {topic} compared to previous questions. 
        Avoid repeating similar concepts or phrasing. This is question number {question_number}.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Function to evaluate answers using Gemini API
def evaluate_answer(question, answer):
    """Evaluate the user's answer using the Gemini API."""
    try:
        prompt = f"""
        Question: {question}
        Answer: {answer}
        Evaluate this answer in terms of correctness, clarity, and depth. Provide constructive feedback.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Function to handle recording and transcription
def record_and_transcribe():
    """Record audio and transcribe it into text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Recording... Speak now!")
        audio = recognizer.listen(source)
        st.write("Recording stopped. Processing...")

        try:
            # Recognize speech using Google Speech Recognition
            recognized_text = recognizer.recognize_google(audio)
            return recognized_text
        except sr.UnknownValueError:
            return "Error: Could not understand the audio."
        except sr.RequestError:
            return "Error: Could not request results from the speech recognition service."

# Sidebar for settings
st.sidebar.write("### Mock Interview Settings")
st.session_state.topic = st.sidebar.selectbox("Select Topic", ["Python", "SQL"])
st.session_state.difficulty = st.sidebar.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])
st.session_state.num_questions = st.sidebar.number_input(
    "How many questions do you want to answer?", 
    min_value=1, 
    max_value=10, 
    value=3
)

# Start the mock interview
if st.sidebar.button("Start Mock Interview"):
    st.session_state.questions = [
        generate_question(st.session_state.topic, st.session_state.difficulty, i + 1)
        for i in range(st.session_state.num_questions)
    ]
    st.session_state.submitted = False  # Reset submission status

# Display all questions and allow the user to answer them
if st.session_state.questions and not st.session_state.submitted:
    st.write("### Mock Interview Questions")
    for i, question in enumerate(st.session_state.questions):
        st.write(f"**Question {i + 1}:** {question}")

        # Start and stop recording buttons
        if st.button(f"Start Recording for Question {i + 1}", key=f"start_button_{i}"):
            st.session_state.recording = True
            recognized_text = record_and_transcribe()
            st.session_state.recognized_texts[i] = recognized_text
            st.session_state.recording = False

        # Display the recognized text
        if i in st.session_state.recognized_texts:
            st.text_area(
                f"Recognized Answer for Question {i + 1}:",
                st.session_state.recognized_texts[i],
                key=f"recognized_text_area_{i}"
            )

    # Submit button to evaluate all answers
    if st.button("Submit All Answers"):
        st.session_state.submitted = True

# Evaluate all answers after submission
if st.session_state.submitted:
    st.write("### Evaluation Results")
    for i, question in enumerate(st.session_state.questions):
        recognized_text = st.session_state.recognized_texts.get(i, "")
        if recognized_text:
            evaluation = evaluate_answer(question, recognized_text)
            st.write(f"**Question {i + 1}:** {question}")
            st.write(f"**Your Answer:** {recognized_text}")
            st.write(f"**Evaluation:** {evaluation}")
            st.write("---")
        else:
            st.warning(f"No answer recorded for Question {i + 1}.")