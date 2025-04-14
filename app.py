from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import requests
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import librosa
import numpy as np
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
import csv
import time
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
def get_all_query1(query):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([query])
    return response.text

# Initialize session states
if 'recognized_text_1' not in st.session_state:
    st.session_state.recognized_text_1 = ""
if 'recognized_text_2' not in st.session_state:
    st.session_state.recognized_text_2 = ""
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'mic_initialized' not in st.session_state:
    st.session_state.mic_initialized = False
if 'tcs_prep' not in st.session_state:
    st.session_state.tcs_prep = False
if 'accenture_prep' not in st.session_state:
    st.session_state.accenture_prep = False
if 'infosys_prep' not in st.session_state:
    st.session_state.infosys_prep = False
if 'wipro_prep' not in st.session_state:
    st.session_state.wipro_prep = False
if 'capgemini_prep' not in st.session_state:
    st.session_state.capgemini_prep = False
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
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

API_KEY2 = os.getenv("JSEARCH_API_KEY")

def get_gemini_response(prompt):
    """Generate a response using Google Gemini API with logging."""
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    start_time = time.time()
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add unique variations each time this prompt is called: {os.urandom(8).hex()}"])
        end_time = time.time()
        
        tokens_generated = len(response.text) // 4 if hasattr(response, 'text') and response.text else 0
        log_api_usage("Gemini_API_Call", 1, tokens_generated, end_time - start_time)
        
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        log_api_usage("Gemini_API_Error", 1, 0, time.time() - start_time)
        return f"Error: {str(e)}"

def get_youtube_transcript(video_id):
    """Fetch transcript from YouTube video with logging."""
    start_time = time.time()
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        tokens = len(text) // 4
        log_api_usage("YouTube_Transcript", 1, tokens, time.time() - start_time)
        return text
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        log_api_usage("YouTube_Transcript_Error", 1, 0, time.time() - start_time)
        return None

def generate_summary_and_insights(transcript):
    """Generate summary and insights from YouTube transcript with logging."""
    if not transcript:
        return "Error: No transcript available."
    
    start_time = time.time()
    prompt = f"""
Analyze the transcript of the video and break it down into key concepts, sections, or steps. 
For each part of the transcript, provide a brief explanation, highlight important points, 
and include relevant images or diagrams that clarify the concepts discussed. 
Organize the transcript into clear sections with visual aids where applicable to better understand the material. 
If there are any technical terms or complex ideas, explain them in simpler terms and use visuals to enhance understanding.
Also show the visual aids diagram.

Transcript:
{transcript}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        tokens = len(response.text) // 4 if hasattr(response, 'text') and response.text else 0
        log_api_usage("Generate_Summary", 1, tokens, time.time() - start_time)
        
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        log_api_usage("Generate_Summary_Error", 1, 0, time.time() - start_time)
        return f"Error: {str(e)}"

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

def generate_question(level, topic, question_num):
    """Generate a unique theoretical interview question with logging."""
    start_time = time.time()
    query = f"Generate a concise, one-line {level} level theoretical interview question (Question {question_num}/3) for a distinct subtopic of {topic} that assesses the candidate's conceptual understanding. Ensure the question is unique, focuses on a different aspect or subtopic than any previous questions in this set, and has not been asked before in this session. Previously asked questions: {', '.join(st.session_state.used_questions) if st.session_state.used_questions else 'None'}."
    question = get_gemini_response(query)
    tokens = len(question) // 4
    log_api_usage(f"Generate_Question_{question_num}", 1, tokens, time.time() - start_time)
    st.session_state.used_questions.append(question)
    return question

# Main App Layout
st.set_page_config(page_title="Data Coders", layout='wide')

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Data Coders</h1>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Input section
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("📋 Job Description:", key="input", height=150)

with col2:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
    if uploaded_file:
        try:
            st.session_state.resume_text = ""
            reader = PdfReader(uploaded_file)
            text_pages = 0
            
            for page in reader.pages:
                if page:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        st.session_state.resume_text += page_text + "\n"
                        text_pages += 1
            
            if text_pages > 0:
                st.success(f"✅ PDF Uploaded Successfully ({text_pages} pages with text).")
            else:
                st.warning("⚠ PDF uploaded but no text content found. Please upload a text-based PDF.")
                st.session_state.resume_text = ""
                
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {str(e)}")
            st.session_state.resume_text = ""

# Button wrapper for logging
def wrap_button_with_logging(button_text, action_key, callback):
    """Wrapper function to add logging to button actions."""
    if st.button(button_text, key=action_key):
        start_time = time.time()
        with st.spinner("⏳ Loading... Please wait"):
            result = callback()
            tokens = len(result) // 4 if isinstance(result, str) else 0
            log_api_usage(action_key, 1, tokens, time.time() - start_time)
            st.session_state.last_result = result
            return result
    return None

# Main Features
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)

if wrap_button_with_logging("📖 Tell Me About the Resume", "resume_info", lambda: (
    get_gemini_response(f"Please review the following resume and provide a detailed evaluation: {st.session_state.resume_text}")
    if st.session_state.resume_text.strip()
    else "⚠ Please upload a valid resume first."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
        st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
    else:
        st.warning(response)

if wrap_button_with_logging("📊 Percentage Match", "percentage_match", lambda: (
    get_gemini_response(f"Evaluate the following resume against this job description and provide a percentage match in first :\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state.resume_text}")
    if st.session_state.resume_text.strip() and input_text.strip()
    else "⚠ Please upload a resume and provide a job description."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
        st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
    else:
        st.warning(response)

learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
if wrap_button_with_logging("🎓 Personalized Learning Path", "learning_path", lambda: (
    get_gemini_response(f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state.resume_text} and also suggest books and other important thing")
    if st.session_state.resume_text.strip() and input_text.strip() and learning_path_duration
    else "⚠ Please upload a resume and provide a job description."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Custom', spaceAfter=12))
        story = [Paragraph(f"Personalized Learning Path ({learning_path_duration})", styles['Title']), Spacer(1, 12)]
        for line in response.split('\n'):
            story.append(Paragraph(line, styles['Custom']))
            story.append(Spacer(1, 12))
        doc.build(story)
        st.download_button(f"💾 Download Learning Path PDF", pdf_buffer.getvalue(), f"learning_path_{learning_path_duration.replace(' ', '_').lower()}.pdf", "application/pdf")
    else:
        st.warning(response)

if wrap_button_with_logging("📝 Generate Updated Resume", "updated_resume", lambda: (
    get_gemini_response(f"Suggest improvements and generate an updated resume for this candidate according to job description, not more than 2 pages:\n{st.session_state.resume_text}")
    if st.session_state.resume_text.strip()
    else "⚠ Please upload a resume first."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
        pdf_file = "updated_resume.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(response.replace('\n', '<br/>'), styles['Normal'])]
        doc.build(story)
        with open(pdf_file, "rb") as f:
            pdf_data = f.read()
        st.download_button(label="📥 Download Updated Resume", data=pdf_data, file_name="Updated_Resume.pdf", mime="application/pdf")
    else:
        st.warning(response)

if wrap_button_with_logging("❓ Generate 30 Interview Questions and Answers", "interview_qa", lambda: (
    get_gemini_response("Generate 30 technical interview questions and their detailed answers according to that job description.")
    if st.session_state.resume_text.strip()
    else "⚠ Please upload a resume first."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
    else:
        st.warning(response)

if wrap_button_with_logging("🚀 Skill Development Plan", "skill_dev_plan", lambda: (
    get_gemini_response(f"Based on the resume and job description, suggest courses, books, and projects to improve the candidate's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state.resume_text}")
    if st.session_state.resume_text.strip() and input_text.strip()
    else "⚠ Please upload a resume first."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
    else:
        st.warning(response)

if wrap_button_with_logging("🎥 Mock Interview Questions", "mock_interview", lambda: (
    get_gemini_response(f"Generate follow-up interview questions based on the resume and job description, simulating a live interview.\n\nJob Description:\n{input_text}\n\nResume:\n{st.session_state.resume_text}")
    if st.session_state.resume_text.strip() and input_text.strip()
    else "⚠ Please upload a resume first."
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("⚠"):
        st.write(response)
    else:
        st.warning(response)

# MNC Preparation Section
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 MNC's preparation</h3>", unsafe_allow_html=True)

# TCS
if wrap_button_with_logging("🎯 TCS Data Science Preparation", "tcs_prep_toggle", lambda: "Toggle TCS Prep"):
    st.session_state.tcs_prep = not st.session_state.tcs_prep

if st.session_state.tcs_prep:
    if wrap_button_with_logging("TCS Main Prep", "tcs_main_prep", lambda: (
        get_gemini_response(f"Based on the candidate's qualifications and resume data, what additional skills and knowledge are needed to secure a Data Science role at TCS?")
        if st.session_state.resume_text.strip()
        else "⚠ Please upload a resume first."
    )):
        response = st.session_state.last_result
        if isinstance(response, str) and not response.startswith("⚠"):
            st.write(response)
        else:
            st.warning(response)

    with st.expander("📂 TCS Additional Resources"):
        if wrap_button_with_logging("📂 TCS Data Science Project Types and Required Skills", "tcs_projects", lambda: (
            get_gemini_response(f"What types of Data Science projects does TCS typically work on, and what additional skills and qualifications from the candidate's resume would align best?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

        if wrap_button_with_logging("🛠 TCS Required Skills", "tcs_skills", lambda: (
            get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at TCS, and how does the candidate's current resume reflect these?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

        if wrap_button_with_logging("💡 TCS Recommendations", "tcs_recommendations", lambda: (
            get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at TCS?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

# Infosys
if wrap_button_with_logging("🎯 Infosys Data Science Preparation", "infosys_prep_toggle", lambda: "Toggle Infosys Prep"):
    st.session_state.infosys_prep = not st.session_state.infosys_prep

if st.session_state.infosys_prep:
    if wrap_button_with_logging("Infosys Main Prep", "infosys_main_prep", lambda: (
        get_gemini_response(f"Based on the candidate's qualifications and resume data, what additional skills and knowledge are needed to secure a Data Science role at Infosys?")
        if st.session_state.resume_text.strip()
        else "⚠ Please upload a resume first."
    )):
        response = st.session_state.last_result
        if isinstance(response, str) and not response.startswith("⚠"):
            st.write(response)
        else:
            st.warning(response)

    with st.expander("📂 Infosys Additional Resources"):
        if wrap_button_with_logging("📂 Infosys Data Science Project Types and Required Skills", "infosys_projects", lambda: (
            get_gemini_response(f"What types of Data Science projects does Infosys typically work on, and what additional skills and qualifications from the candidate's resume would align best?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

        if wrap_button_with_logging("🛠 Infosys Required Skills", "infosys_skills", lambda: (
            get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at Infosys, and how does the candidate's current resume reflect these?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

        if wrap_button_with_logging("💡 Infosys Recommendations", "infosys_recommendations", lambda: (
            get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at Infosys?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

# Wipro
if wrap_button_with_logging("🎯 Wipro Data Science Preparation", "wipro_prep_toggle", lambda: "Toggle Wipro Prep"):
    st.session_state.wipro_prep = not st.session_state.wipro_prep

if st.session_state.wipro_prep:
    if wrap_button_with_logging("Wipro Main Prep", "wipro_main_prep", lambda: (
        get_gemini_response(f"Based on the candidate's qualifications and resume data, what additional skills and knowledge are needed to secure a Data Science role at Wipro?")
        if st.session_state.resume_text.strip()
        else "⚠ Please upload a resume first."
    )):
        response = st.session_state.last_result
        if isinstance(response, str) and not response.startswith("⚠"):
            st.write(response)
        else:
            st.warning(response)

    with st.expander("📂 Wipro Additional Resources"):
        if wrap_button_with_logging("📂 Wipro Data Science Project Types and Required Skills", "wipro_projects", lambda: (
            get_gemini_response(f"What types of Data Science projects does Wipro typically work on, and what additional skills and qualifications from the candidate's resume would align best?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

        if wrap_button_with_logging("🛠 Wipro Required Skills", "wipro_skills", lambda: (
            get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at Wipro, and how does the candidate's current resume reflect these?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

        if wrap_button_with_logging("💡 Wipro Recommendations", "wipro_recommendations", lambda: (
            get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at Wipro?")
            if st.session_state.resume_text.strip()
            else "⚠ Please upload a resume first."
        )):
            response = st.session_state.last_result
            if isinstance(response, str) and not response.startswith("⚠"):
                st.write(response)
            else:
                st.warning(response)

# DSA Section
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 DSA Questions for Data Science</h3>", unsafe_allow_html=True)

level = st.selectbox("📚 Select Difficulty Level:", ["Easy", "Intermediate", "Advanced"])

if wrap_button_with_logging(f"📝 Generate {level} DSA Questions (Data Science)", "dsa_questions", lambda: (
    get_gemini_response(f"I have a Data Structures and Algorithms (DSA) question related to Data Science. Based on its difficulty {level}, provide a well-structured solution. Explain the approach in a simple and easy-to-understand way, using analogies or step-by-step breakdowns where necessary. If applicable, include diagrams or visual representations to enhance clarity. Please generate solutions for 10 different questions, ensuring variety in topics relevant to DSA in Data Science.")
)):
    response = st.session_state.last_result
    st.write(response)

# Define categories
categories = {
    "Data Structures": [
        "Arrays (1D, 2D, Dynamic arrays)",
        "Linked Lists (Singly, Doubly, Circular)",
        "Stacks",
        "Queues (Simple, Circular, Deque, Priority Queue)",
        "Hashing & Hash Tables",
        "Trees (Binary Tree, BST, AVL, B-Trees, Trie)",
        "Graphs (Adjacency List, Adjacency Matrix, DFS, BFS)",
        "Heaps (Min-Heap, Max-Heap)",
        "Segment Trees & Fenwick Trees",
        "Disjoint Set (Union-Find)"
    ],
    "Algorithms": [
        "Sorting Algorithms (Bubble, Selection, Insertion, Merge, Quick, Counting, Radix)",
        "Searching Algorithms (Binary Search, Linear Search, Ternary Search)",
        "Recursion & Backtracking",
        "Dynamic Programming (Knapsack, LCS, Fibonacci, Coin Change)",
        "Greedy Algorithms (Huffman Encoding, Kruskal's Algorithm, Prim's Algorithm)",
        "Graph Algorithms (Dijkstra, Floyd-Warshall, Bellman-Ford, A*)",
        "Tree Algorithms (DFS, BFS, Lowest Common Ancestor, Segment Trees)",
        "Bit Manipulation",
        "String Algorithms (KMP, Rabin-Karp, Z-Algorithm, Manacher's Algorithm)"
    ]
}

# First select category
category = st.selectbox("Select Category", list(categories.keys()))

# Then select topic based on category
topic = st.selectbox("Select Topic", categories[category])

if wrap_button_with_logging(f"📖 Teach me {topic} with Case Studies", "teach_topic", lambda: (
    get_gemini_response(f"Teach me {topic} for data science with real-world case studies and examples. Provide a brief case study on how this {topic} is used in companies, explaining its practical applications and implementation. Highlight how it helps businesses improve efficiency, decision-making, or customer experience. Include specific industries or organizations that have successfully used this {topic}, along with a simplified explanation of the process involved.Make the explanation as simple as possible, adding 2-3 more lines for better clarity. Also, provide relevant code snippets demonstrating the topic's use cases, along with a brief and easy-to-understand explanation of how the code works. If possible, include a visual representation (such as a diagram, flowchart, or graph) to illustrate key concepts and improve understanding.")
)):
    response = st.session_state.last_result
    st.write(response)

# Interview Questions Section
st.markdown("---")
question_category = st.selectbox("❓ Select Question Category:", ["Python", "Machine Learning", "Deep Learning", "Docker", "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"])

if wrap_button_with_logging(f"📝 Generate 30 {question_category} Interview Questions", "category_questions", lambda: (
    get_gemini_response(f"Generate 30 {question_category} interview questions and detailed answers")
)):
    response = st.session_state.last_result
    st.write(response)

# Job Search Section
st.subheader("Click on a company to view job description:")
companies = ["TCS", "Wipro", "Infosys", "Accenture", "Cognizant"]

def fetch_jobs(company):
    if not API_KEY2:
        st.warning("JSEARCH_API_KEY not configured")
        return []
    
    start_time = time.time()
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"{company} Data Scientist", "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": API_KEY2,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            data = response.json().get("data", [])
            tokens = len(str(data)) // 4
            log_api_usage(f"Fetch_Jobs_{company}", 1, tokens, time.time() - start_time)
            return data
        log_api_usage(f"Fetch_Jobs_{company}_Error", 1, 0, time.time() - start_time)
        return []
    except Exception as e:
        st.error(f"Error fetching jobs: {str(e)}")
        log_api_usage(f"Fetch_Jobs_{company}_Error", 1, 0, time.time() - start_time)
        return []

selected_company = None
for company in companies:
    if wrap_button_with_logging(company, f"fetch_jobs_{company}", lambda c=company: c):
        selected_company = st.session_state.last_result

if selected_company:
    st.subheader(f"Job Listings at {selected_company}")
    jobs = fetch_jobs(selected_company)
    if jobs:
        for job in jobs:
            st.markdown(f"### {job.get('job_title', 'Job Title Not Available')}")
            st.write(f"Company: {job.get('employer_name', 'N/A')}")
            st.write(f"Location: {job.get('job_city', 'Unknown')}, {job.get('job_country', 'Unknown')}")
            st.write(f"Description: {job.get('job_description', 'No description available.')}")
            st.markdown(f"[Apply Here]({job.get('job_apply_link', '#')})")
            st.write("---")
    else:
        st.write("No job listings found. Try again later!")

# Voice Input Section
st.subheader("Voice Input")
audio_dict = mic_recorder(start_prompt="Click to Speak", stop_prompt="Stop Recording", key="mic_general")
recognized_text = process_audio(audio_dict, "recognized_text_1", 0)
if recognized_text:
    start_time = time.time()
    response = get_all_query1(recognized_text)
    tokens = len(response) // 4
    log_api_usage("Voice_Input_Response", 1, tokens, time.time() - start_time)
    st.subheader("Response:")
    st.write(response)

# Text input fallback
query = st.text_input("HelpDesk", key="text_query")
if wrap_button_with_logging("Ask", "ask_query", lambda: (
    get_gemini_response(query) if query else "Please enter or speak a query!"
)):
    response = st.session_state.last_result
    if isinstance(response, str) and not response.startswith("Please"):
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning(response)

# Mock Interview System
st.title("AI-Powered Mock Interview System")

topic = st.radio("Select Topic:", ("Python", "SQL"), key="mock_topic")
level = st.radio("Select Difficulty:", ("Easy", "Intermediate", "Hard"), key="mock_level", index=["Easy", "Intermediate", "Hard"].index(st.session_state.difficulty))

if wrap_button_with_logging("Start Interview", "start_interview", lambda: "Start"):
    st.session_state.questions = [
        generate_question(st.session_state.difficulty, topic, 1),
        generate_question(st.session_state.difficulty, topic, 2),
        generate_question(st.session_state.difficulty, topic, 3)
    ]
    st.session_state.answers = []
    st.session_state.current_question_index = 0
    st.session_state.started = True
    st.session_state.interview_complete = False
    st.session_state.audio_dict_mock = None
    st.rerun()

if st.session_state.started and not st.session_state.interview_complete:
    if st.session_state.current_question_index < 3:
        current_question = st.session_state.questions[st.session_state.current_question_index]
        st.write(f"**Question {st.session_state.current_question_index + 1}/3:** {current_question}")

        st.session_state.audio_dict_mock = mic_recorder(
            start_prompt=f"Click to Speak Your Answer for Question {st.session_state.current_question_index + 1}",
            stop_prompt="Stop Recording",
            key=f"mic_mock_interview_{st.session_state.current_question_index}"
        )

        if st.session_state.audio_dict_mock:
            recognized_text_mock = process_audio(st.session_state.audio_dict_mock, "recognized_text_2", st.session_state.current_question_index)
            if recognized_text_mock:
                st.session_state.answers.append(recognized_text_mock)
                start_time = time.time()
                evaluation = get_gemini_response(f"Evaluate this answer in terms of correctness, clarity, and depth for the question '{current_question}': {recognized_text_mock}")
                tokens = len(evaluation) // 4
                log_api_usage(f"Evaluate_Answer_Q{st.session_state.current_question_index + 1}", 1, tokens, time.time() - start_time)
                st.subheader(f"Evaluation for Question {st.session_state.current_question_index + 1}:")
                st.write(evaluation)

                if "good" in evaluation.lower() and st.session_state.difficulty != "Hard":
                    st.session_state.difficulty = "Intermediate" if st.session_state.difficulty == "Easy" else "Hard"
                elif "poor" in evaluation.lower() and st.session_state.difficulty != "Easy":
                    st.session_state.difficulty = "Easy" if st.session_state.difficulty == "Hard" else "Intermediate"

                st.session_state.current_question_index += 1
                st.session_state.audio_dict_mock = None
                if st.session_state.current_question_index < 3:
                    st.rerun()
                else:
                    st.session_state.interview_complete = True
                    st.rerun()

if st.session_state.interview_complete and len(st.session_state.answers) == 3:
    st.subheader("Interview Completed!")
    combined_answers = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers))])
    start_time = time.time()
    feedback = get_gemini_response(f"Provide overall feedback for these 3 question-answer pairs and suggest improvements:\n{combined_answers}")
    tokens = len(feedback) // 4
    log_api_usage("Overall_Feedback", 1, tokens, time.time() - start_time)
    st.subheader("Overall Feedback:")
    st.write(feedback)

    if wrap_button_with_logging("Restart Interview", "restart_interview", lambda: "Restart"):
        st.session_state.started = False
        st.session_state.interview_complete = False
        st.session_state.current_question_index = 0
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.audio_dict_mock = None
        st.rerun()

# YouTube Video Analysis Section
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>YouTube Video Analyzer</h1>", unsafe_allow_html=True)
st.markdown("---")

youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    try:
        if "v=" in youtube_link:
            video_id = youtube_link.split("v=")[1].split("&")[0]
        else:
            video_id = youtube_link

        with st.spinner("⏳ Fetching transcript..."):
            transcript = get_youtube_transcript(video_id)
        
        if transcript:
            if wrap_button_with_logging("Generate Insights", "youtube_insights", lambda: generate_summary_and_insights(transcript)):
                insights = st.session_state.last_result
                st.subheader("Summary and Insights:")
                st.write(insights)
                st.download_button(
                    label="💾 Download Insights",
                    data=insights,
                    file_name="youtube_insights.txt",
                    mime="text/plain"
                )
        else:
            st.warning("⚠ Failed to fetch transcript. Please check the video link.")
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")

