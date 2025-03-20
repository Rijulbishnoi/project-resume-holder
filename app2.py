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

if 'recognized_text_1' not in st.session_state:
    st.session_state.recognized_text_1 = ""  # For the first "Click to Speak"

if 'recognized_text_2' not in st.session_state:
    st.session_state.recognized_text_2 = ""

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
    """Generate a response using Google Gemini API."""
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add unique variations each time this prompt is called: {os.urandom(8).hex()}"])
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return f"Error: {str(e)}"
def get_gemini_response1(prompt):
    """Generate a response using Google Gemini API."""
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    
    try:
        # Call Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add unique variations each time this prompt is called: {os.urandom(8).hex()}"])
        
        # Check if the response contains valid text
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return f"Error: {str(e)}"

st.set_page_config(page_title="A5 ATS Resume Expert", layout='wide')

# Header with a fresh style
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>MY PERSONAL ATS</h1>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Input section with better layout
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("📋 Job Description:", key="input", height=150)

uploaded_file = None
resume_text = ""
with col2:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
    if uploaded_file:
        st.success("✅ PDF Uploaded Successfully.")
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                if page and page.extract_text():
                    resume_text += page.extract_text()
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {str(e)}")

# Always visible buttons styled
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 Quick Actions</h3>", unsafe_allow_html=True)

# Full-width response area
response_container = st.container()

# Ensure response container takes full width
# with st.expander("📋 Response", expanded=True):
#     response_container = st.empty()

# Button actions
if st.button("📖 Tell Me About the Resume"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Please review the following resume and provide a detailed evaluation: {resume_text}")
            st.write(response)
            st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
        else:
            st.warning("⚠ Please upload a valid resume first.")

if st.button("📊 Percentage Match"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text:
            response = get_gemini_response(f"Evaluate the following resume against this job description and provide a percentage match in first :\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
            st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
        else:
            st.warning("⚠ Please upload a resume and provide a job description.")

learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
if st.button("🎓 Personalized Learning Path"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text and learning_path_duration:
            response = get_gemini_response(f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text} and also suggest books and other important thing")
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
            st.warning("⚠ Please upload a resume and provide a job description.")

if st.button("📝 Generate Updated Resume"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Suggest improvements and generate an updated resume for this candidate according to job description, not more than 2 pages:\n{resume_text}")
            st.write(response)

            # Convert response to PDF
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet

            pdf_file = "updated_resume.pdf"
            doc = SimpleDocTemplate(pdf_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph(response.replace('\n', '<br/>'), styles['Normal'])]
            doc.build(story)

            # Read PDF as binary
            with open(pdf_file, "rb") as f:
                pdf_data = f.read()

            # Download button for PDF
            st.download_button(label="📥 Download Updated Resume", data=pdf_data, file_name="Updated_Resume.pdf", mime="application/pdf")
        else:
            st.warning("⚠ Please upload a resume first.")


if st.button("❓ Generate 30 Interview Questions and Answers"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response("Generate 30 technical interview questions and their detailed answers according to that job description.")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")


if st.button("🚀 Skill Development Plan"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text:
            response = get_gemini_response(f"Based on the resume and job description, suggest courses, books, and projects to improve the candidate's weak or missing skills.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")

if st.button("🎥 Mock Interview Questions"):
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text and input_text:
            response = get_gemini_response(f"Generate follow-up interview questions based on the resume and job description, simulating a live interview.\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")


st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 MNC's preparation</h3>", unsafe_allow_html=True)









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

# TCS
if st.button("🎯 TCS Data Science Preparation"):
    st.session_state.tcs_prep = not st.session_state.tcs_prep

if st.session_state.tcs_prep:
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Based on the candidate's qualifications and resume data, what additional skills and knowledge are needed to secure a Data Science role at TCS?")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")

    with st.expander("📂 TCS Additional Resources"):
        if st.button("📂 TCS Data Science Project Types and Required Skills", key="tcs_projects"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"What types of Data Science projects does TCS typically work on, and what additional skills and qualifications from the candidate's resume would align best?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("🛠 TCS Required Skills", key="tcs_skills"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at TCS, and how does the candidate's current resume reflect these?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("💡 TCS Recommendations", key="tcs_recommendations"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at TCS?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

# Infosys
if st.button("🎯 Infosys Data Science Preparation"):
    st.session_state.infosys_prep = not st.session_state.infosys_prep

if st.session_state.infosys_prep:
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Based on the candidate's qualifications and resume data, what additional skills and knowledge are needed to secure a Data Science role at Infosys?")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")

    with st.expander("📂 Infosys Additional Resources"):
        if st.button("📂 Infosys Data Science Project Types and Required Skills", key="infosys_projects"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"What types of Data Science projects does Infosys typically work on, and what additional skills and qualifications from the candidate's resume would align best?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("🛠 Infosys Required Skills", key="infosys_skills"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at Infosys, and how does the candidate's current resume reflect these?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("💡 Infosys Recommendations", key="infosys_recommendations"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at Infosys?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

# Wipro
if st.button("🎯 Wipro Data Science Preparation"):
    st.session_state.wipro_prep = not st.session_state.wipro_prep

if st.session_state.wipro_prep:
    with st.spinner("⏳ Loading... Please wait"):
        if resume_text:
            response = get_gemini_response(f"Based on the candidate's qualifications and resume data, what additional skills and knowledge are needed to secure a Data Science role at Wipro?")
            st.write(response)
        else:
            st.warning("⚠ Please upload a resume first.")

    with st.expander("📂 Wipro Additional Resources"):
        if st.button("📂 Wipro Data Science Project Types and Required Skills", key="wipro_projects"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"What types of Data Science projects does Wipro typically work on, and what additional skills and qualifications from the candidate's resume would align best?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("🛠 Wipro Required Skills", key="wipro_skills"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"What key technical and soft skills are needed for a Data Science role at Wipro, and how does the candidate's current resume reflect these?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")

        if st.button("💡 Wipro Recommendations", key="wipro_recommendations"):
            with st.spinner("⏳ Loading... Please wait"):
                if resume_text:
                    response = get_gemini_response(f"Based on the candidate's resume, what specific areas should they focus on to strengthen their chances of getting a Data Science role at Wipro?")
                    st.write(response)
                else:
                    st.warning("⚠ Please upload a resume first.")




st.markdown("---")


st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠 DSA Questions for Data Science</h3>", unsafe_allow_html=True)

 # Main DSA Questions button
level = st.selectbox("📚 Select Difficulty Level:", ["Easy", "Intermediate", "Advanced"])

if st.button(f"📝 Generate {level} DSA Questions (Data Science)"):
    with st.spinner("⏳ Loading... Please wait"):
        response = get_gemini_response(f"I have a Data Structures and Algorithms (DSA) question related to Data Science. Based on its difficulty {level}, provide a well-structured solution. Explain the approach in a simple and easy-to-understand way, using analogies or step-by-step breakdowns where necessary. If applicable, include diagrams or visual representations to enhance clarity. Please generate solutions for 10 different questions, ensuring variety in topics relevant to DSA in Data Science.")
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
        "Greedy Algorithms (Huffman Encoding, Kruskal’s Algorithm, Prim’s Algorithm)",
        "Graph Algorithms (Dijkstra, Floyd-Warshall, Bellman-Ford, A*)",
        "Tree Algorithms (DFS, BFS, Lowest Common Ancestor, Segment Trees)",
        "Bit Manipulation",
        "String Algorithms (KMP, Rabin-Karp, Z-Algorithm, Manacher’s Algorithm)"
    ]
}

# First select category
category = st.selectbox("Select Category", list(categories.keys()))

# Then select topic based on category
topic = st.selectbox("Select Topic", categories[category])


if st.button(f"📖 Teach me {topic} with Case Studies"):
    with st.spinner("⏳ Gathering resources... Please wait"):
        case_study_response = get_gemini_response(f"Teach me {topic} for data science with real-world case studies and examples. Provide a brief case study on how this {topic} is used in companies, explaining its practical applications and implementation. Highlight how it helps businesses improve efficiency, decision-making, or customer experience. Include specific industries or organizations that have successfully used this {topic}, along with a simplified explanation of the process involved.Make the explanation as simple as possible, adding 2-3 more lines for better clarity. Also, provide relevant code snippets demonstrating the topic’s use cases, along with a brief and easy-to-understand explanation of how the code works. If possible, include a visual representation (such as a diagram, flowchart, or graph) to illustrate key concepts and improve understanding.This ensures you get a complete answer with real-world applications, easy explanations, practical code, and visuals when needed! 🚀.")
        st.write(case_study_response)




st.markdown("---")


question_category = st.selectbox("❓ Select Question Category:", ["Python", "Machine Learning", "Deep Learning", "Docker", "Data Warehousing", "Data Pipelines", "Data Modeling", "SQL"])

if st.button(f"📝 Generate 30 {question_category} Interview Questions"):
    with st.spinner("⏳ Loading... Please wait"):
        response = get_gemini_response(f"Generate 30 {question_category} interview questions and detailed answers")
        st.write(response)
        st.write(response)
st.subheader("Click on a company to view job description:")

# List of companies
companies = ["TCS", "Wipro", "Infosys", "Accenture", "Cognizant"]

# Function to fetch jobs using JSearch API
def fetch_jobs(company):
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"{company} Data Scientist", "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": API_KEY2,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        jobs = response.json().get("data", [])
        return jobs
    else:
        return []

# Create buttons for each company
selected_company = None
for company in companies:
    if st.button(company):
        selected_company = company

# Display job description if a company is selected
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
    st.subheader("Voice Input")
    st.markdown(
    """
    <style>
    .top-right {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Audio recording and processing
audio_dict = mic_recorder(start_prompt="Click to Speak", stop_prompt="Stop Recording", key="mic_1")

# Create a container for the top-right UI
with st.container():
    st.markdown('<div class="top-right">', unsafe_allow_html=True)

    # Display the audio recorder
    # First "Click to Speak" Section
if audio_dict and "bytes" in audio_dict:
    st.success("Audio Recorded Successfully!")
    try:
        # Convert recorded audio to WAV format using pydub
        audio_bytes = audio_dict["bytes"]
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")  # Assuming webm format
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")  # Export to WAV format
        wav_buffer.seek(0)

        # Use speech_recognition to process the WAV file
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
            audio = recognizer.record(source)  # Record the entire audio file
            recognized_text = recognizer.recognize_google(audio)  # Perform speech recognition

            # Store the recognized text in session state
            st.session_state.recognized_text_1 = recognized_text

            # Display the recognized text with a unique key
            st.text_area("Recognized Text:", st.session_state.recognized_text_1, key="recognized_text_area_1")
            query1 = st.session_state.recognized_text_1  # Set recognized text as query

            if query1:
                response = get_gemini_response1(query1)
                st.subheader("Response:")
                st.write(response)

    except sr.UnknownValueError as e:
        st.warning(f"Could not understand the audio. Please try again in a quiet environment and speak clearly.{e}")
    except sr.RequestError:
        st.warning("Error connecting to the speech recognition service. Please check your internet connection.")
    except Exception as e:
        st.warning(f"An error occurred: {e}")
    # Text input for query
    query= st.text_input("HelpDesk", key="text_query")

    # Button to submit the query
    if st.button("Ask") or query:
        if query:
            response = get_gemini_response(query)
            st.subheader("Response:")
            st.write(response)
        else:
            st.warning("Please enter or speak a query!")

    st.markdown('</div>', unsafe_allow_html=True)

def generate_question(level, topic):
    """Generate a structured interview question."""
    query = f"Ask a concise, one-line {level} level theoretical {topic} interview question that assesses the candidate’s conceptual understanding."
    return get_gemini_response(query)

# Streamlit UI
st.title("AI-Powered Mock Interview System")

# User selects topic & difficulty level
topic = st.radio("Select Topic:", ("Python", "SQL"))
level = st.radio("Select Difficulty:", ("Easy", "Intermediate", "Hard"))

if 'answers' not in st.session_state:
    st.session_state.answers = []

if 'question' not in st.session_state:
    st.session_state.question = None  # No question until interview starts

# Button to start the interview session
if st.button("Start Interview"):
    st.session_state.question = generate_question(level, topic)
    st.session_state.started = True  # Mark interview as started
    st.rerun()

# Display the first question only after clicking "Start Interview"
# Imports and setup
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io

# Session state variables
if 'recognized_text_1' not in st.session_state:
    st.session_state.recognized_text_1 = ""  # For the first "Click to Speak"

if 'recognized_text_2' not in st.session_state:
    st.session_state.recognized_text_2 = ""  # For the mock interview

# Streamlit UI
st.title("My App")

# First "Click to Speak" Section
audio_dict = mic_recorder(start_prompt="Click to Speak", stop_prompt="Stop Recording", key="mic_1")

if audio_dict and "bytes" in audio_dict:
    st.success("Audio Recorded Successfully!")
    try:
        # Convert recorded audio to WAV format using pydub
        audio_bytes = audio_dict["bytes"]
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")  # Assuming webm format
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")  # Export to WAV format
        wav_buffer.seek(0)

        # Use speech_recognition to process the WAV file
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
            audio = recognizer.record(source)  # Record the entire audio file
            recognized_text = recognizer.recognize_google(audio)  # Perform speech recognition

            # Store the recognized text in session state
            st.session_state.recognized_text_1 = recognized_text

            # Display the recognized text with a unique key
            st.text_area("Recognized Text:", st.session_state.recognized_text_1, key="recognized_text_area_1")
            query1 = st.session_state.recognized_text_1  # Set recognized text as query

            if query1:
                response = get_gemini_response1(query1)
                st.subheader("Response:")
                st.write(response)

    except sr.UnknownValueError as e:
        st.warning(f"Could not understand the audio. Please try again in a quiet environment and speak clearly.{e}")
    except sr.RequestError:
        st.warning("Error connecting to the speech recognition service. Please check your internet connection.")
    except Exception as e:
        st.warning(f"An error occurred: {e}")

# Mock Interview Section
if 'started' in st.session_state and st.session_state.started:
    st.write(f"**Question:** {st.session_state.question}")

    # Button to start speaking for the mock interview
    if st.button("Click to Speak Your Answer", key="mock_interview_speak_button"):
        audio_dict_mock = mic_recorder(start_prompt="Click to Speak", stop_prompt="Stop Recording", key="mic_mock_interview")

        if audio_dict_mock and "bytes" in audio_dict_mock:
            st.success("Audio Recorded Successfully!")
            try:
                # Convert recorded audio to WAV format using pydub
                audio_bytes = audio_dict_mock["bytes"]
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")  # Assuming webm format
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")  # Export to WAV format
                wav_buffer.seek(0)

                # Use speech_recognition to process the WAV file
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_buffer) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
                    audio = recognizer.record(source)  # Record the entire audio file
                    recognized_text_mock = recognizer.recognize_google(audio)  # Perform speech recognition

                    # Store the recognized text in session state
                    st.session_state.recognized_text_2 = recognized_text_mock

                    # Display the recognized text with a unique key
                    st.text_area("Recognized Answer:", st.session_state.recognized_text_2, key="recognized_answer_area_mock")

                    # Evaluate the answer
                    def evaluate_answer(answer):
                        """Evaluate the answer and return feedback."""
                        return get_gemini_response(f"Evaluate this answer in terms of correctness, clarity, and depth: {answer}")

                    evaluation = evaluate_answer(st.session_state.recognized_text_2)
                    st.subheader("Evaluation:")
                    st.write(evaluation)

                    # Adjust difficulty based on evaluation
                    if "good" in evaluation.lower():
                        level = "Intermediate" if level == "Easy" else "Hard"
                    elif "poor" in evaluation.lower():
                        level = "Easy"

                    # Generate the next question
                    st.session_state.question = generate_question(level, topic)
                    st.rerun()

            except sr.UnknownValueError:
                st.warning("Could not understand the audio. Please try again in a quiet environment.")
            except sr.RequestError:
                st.warning("Error connecting to the speech recognition service.")
            except Exception as e:
                st.warning(f"An error occurred: {e}")

    # Provide overall feedback after 3-4 answers
    if len(st.session_state.answers) >= 3:
        combined_answers = "\n".join(st.session_state.answers)
        feedback = get_gemini_response(f"Provide an overall feedback for these answers and suggest improvements: {combined_answers}")
        st.subheader("Overall Feedback:")
        st.write(feedback)

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
        full_transcript = " ".join([entry['text'] for entry in transcript])
        return full_transcript
    except Exception as e:
        st.error(f"Failed to fetch transcript: {str(e)}")
        return None

# Function to generate summary and insights using Gemini API
def generate_summary_and_insights(transcript):
    if not transcript:
        return "Error: No transcript available."
    
    prompt = f"""

Analyze the transcript of the video and break it down into key concepts, sections, or steps. For each part of the transcript, provide a brief explanation, highlight important points, and include relevant images or diagrams that clarify the concepts discussed. Organize the transcript into clear sections with visual aids where applicable to better understand the material. If there are any technical terms or complex ideas, explain them in simpler terms and use visuals to enhance understanding.
.Also show the visual aids diagram


    Transcript:
    {transcript}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI for YouTube Video Analysis

    
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>YouTube Video Analyzer</h1>", unsafe_allow_html=True)
st.markdown("---")

    # Input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
        try:
            # Extract video ID from the link
            if "v=" in youtube_link:
                video_id = youtube_link.split("v=")[1].split("&")[0]
            else:
                video_id = youtube_link  # Assume the user directly entered the video ID
            
            # Fetch transcript
            with st.spinner("⏳ Fetching transcript..."):
                transcript = get_youtube_transcript(video_id)
            
            if transcript:
                
                
                # Generate summary and insights
                with st.spinner("⏳ Generating summary and insights..."):
                    insights = generate_summary_and_insights(transcript)
                
                st.subheader("Summary and Insights:")
                st.write(insights)
                
                # Download insights as a text file
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