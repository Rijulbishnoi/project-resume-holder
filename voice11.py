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
    """Generate a response using Google Gemini API with session ID for uniqueness."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([f"Session ID: {session_id}", input_text, pdf_content[0], prompt])
    return response.text

def get_gemini_response_question(prompt, session_id):
    """Generate a response for interview questions using Google Gemini API with session ID."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([f"Session ID: {session_id}", prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    """Convert first page of uploaded PDF to an image and encode as base64."""
    if uploaded_file is not None:
        uploaded_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(uploaded_file.read())  # Removed poppler_path
        first_page = images[0]

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()  # Encode to base64
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
    pdf_buffer.write(pdf.output(dest='S').encode('latin1'))  # Properly encode
    pdf_buffer.seek(0)

    return pdf_buffer


def get_pdf_download_link(pdf_buffer, filename):
    """Create a Streamlit download link for the PDF."""
    b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{filename}">Download PDF</a>'
    return href

# Streamlit UI
st.set_page_config(page_title="Your Career helper")
st.header("MY A5 PERSONAL ATS")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

if uploaded_file:
    st.success("PDF Uploaded Successfully.")
else:
    pdf_content = None
def get_all_query(query):
    """Generate a response for interview questions using Google Gemini API with session ID."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([query])
    return response.text
import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import io

# Function to recognize speech from audio bytes
def recognize_speech(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        # Convert audio bytes to a format that speech_recognition can process
        audio_buffer = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_buffer) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the speech."
    except sr.RequestError:
        return "Error connecting to speech recognition service."

# Function to get response from Gemini API
def get_all_query(query):
    # Replace this with your actual Gemini API call
    return f"Response to: {query}"

# Streamlit UI

st.header("Speak and Get Response 🎤")

# Browser-based audio recorder
st.subheader("Click to Speak")
audio_dict = mic_recorder(start_prompt="🎤 Click to Speak", stop_prompt="⏹ Stop Recording", key="mic")

# Process recorded audio
if audio_dict and "bytes" in audio_dict:
    st.success("Audio Recorded Successfully!")
    
    # Recognize speech from the recorded audio
    query = recognize_speech(audio_dict["bytes"])
    st.text_area("Recognized Text:", query)  # Show converted speech
    
    # Get response from Gemini API
    if query and "sorry" not in query.lower():  # Skip if speech was not understood
        response = get_all_query(query)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please try speaking again.")
# Buttons for different actions
submit_python = st.button("Python Questions")
submit_ml = st.button("Machine Learning Questions")
submit_dl = st.button("Deep Learning Questions")
submit_docker = st.button("Docker Questions")
submit_resume = st.button("Tell Me About the Resume")
submit_match = st.button("Percentage Match")
submit_learning = st.button("Personalized Learning Path")
submit_enhance = st.button("Enhance Resume Score")

input_prompt_resume = """
You are an experienced HR with tech expertise in Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, or Data Analysis.
Your task is to review the provided resume against the job description for these roles.
Please evaluate the candidate's profile, highlighting strengths and weaknesses in relation to the specified job role.
"""

input_prompt_match= """
You are a skilled ATS (Applicant Tracking System) scanner with expertise in Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, and Data Analysis.
Your task is to evaluate the resume against the job description. Provide:
1. The percentage match.
2. Keywords missing.
3. Final evaluation.
"""

input_prompt_learning= """
You are an experienced learning coach and technical expert. Create a 6-month personalized study plan for an individual aiming to excel in [Job Role], 
focusing on the skills, topics, and tools specified in the provided job description. Ensure the study plan includes:
- A list of topics and tools for each month.
- Suggested resources (books, online courses, documentation).
- Recommended practical exercises or projects.
- Periodic assessments or milestones.
- Tips for real-world applications.
"""
input_prompt_enhance=f"""
"I have a resume that needs to be optimized to match a specific {input_text}. The goal is to improve its ATS score, highlight relevant skills, and increase the chances of getting shortlisted for an interview.

The job description and resume are already uploaded. Your task is to:

Extract Key Skills & Keywords 1. Identify important skills, technologies, and keywords from the job description and ensure they are well-represented in the resume.
Optimize Work Experience & Achievements 2. Rewrite bullet points to emphasize impact, metrics, and accomplishments instead of just listing tasks.
Enhance Project Descriptions 3. Ensure projects showcase relevant technologies, methodologies, and results that align with the job role.
Improve Resume Formatting & Structure 4. Keep the document clear, concise, and visually appealing while ensuring it remains professional.
Strengthen Summary & Certifications 5. Update the summary to align with the role and highlight relevant certifications, courses, and achievements.
Once optimized, generate a new resume in PDF format and provide it for download while ensuring factual accuracy and job relevance"*
"""

input_prompt_python = """
As an HR specializing in technical hiring, I need a comprehensive set of Python interview questions to evaluate candidates' core Python expertise. 

Generate 30 Python questions categorized into three levels: 10 basic, 10 intermediate, and 10 advanced.

Ensure a balanced mix of:
- **Theoretical questions** covering Python fundamentals, data types, OOP, exception handling, memory management, iterators, generators, and metaprogramming.
- **Logical problem-solving questions** involving control flow, recursion, data structures, algorithm implementation, and code debugging.

For each question:
1. Provide the question.
2. Give a **detailed answer** with a proper explanation.
3. Include **code examples and best practices** where relevant.
4. Ensure **clear, structured, and in-depth responses** without any word limits.

Provide all 30 questions with their corresponding answers and explanations.
"""
input_prompt_ml = """
As an HR specializing in technical hiring, I need a comprehensive set of Machine Learning interview questions to evaluate candidates' expertise.

Generate 30 Machine Learning questions categorized into three levels: 10 basic, 10 intermediate, and 10 advanced.

Ensure a balanced mix of:
- **Theoretical questions** covering ML fundamentals, supervised vs. unsupervised learning, feature engineering, evaluation metrics, overfitting, regularization, ensemble learning, and model selection.
- **Practical problem-solving questions** involving data preprocessing, algorithm implementation, hyperparameter tuning, optimization techniques, and debugging ML pipelines.

For each question:
1. Provide the question.
2. Give a **detailed answer** with a full explanation.
3. Include **mathematical formulas, equations, and real-world examples** where needed.
4. Ensure **clear, structured, and in-depth responses** without any word limits.

Provide all 30 questions with their corresponding answers and explanations.
"""
input_prompt_dl = """
As an HR specializing in hiring Deep Learning engineers, I need a structured set of interview questions to assess candidates' expertise in neural networks and deep learning concepts.

Generate 30 Deep Learning questions categorized into three levels: 10 basic, 10 intermediate, and 10 advanced.

Ensure a balanced mix of:
- **Theoretical questions** covering perceptrons, activation functions, loss functions, backpropagation, CNNs, RNNs, LSTMs, Transformers, GANs, and attention mechanisms.
- **Logical problem-solving questions** involving neural network implementation, optimizing architectures, hyperparameter tuning, vanishing gradients, and debugging deep learning models.

For each question:
1. Provide the question.
2. Give a **detailed answer** with an in-depth explanation.
3. Include **diagrams, code snippets, and practical applications** where relevant.
4. Ensure **clear, structured, and detailed responses** without any word limits.

Provide all 30 questions with their corresponding answers and explanations.
"""
input_prompt_docker = """
As an HR specializing in hiring for Data Science and DevOps roles, I need a structured set of Docker interview questions tailored for ML and AI applications.

Generate 30 Docker questions categorized into three levels: 10 basic, 10 intermediate, and 10 advanced.

Ensure a balanced mix of:
- **Theoretical questions** covering Docker fundamentals, containers vs. VMs, Dockerfiles, images, volumes, networking, orchestration, and security best practices.
- **Practical problem-solving questions** involving containerizing ML models, optimizing Docker images for AI workloads, handling dependencies, using Docker Compose, and debugging containerized applications.

For each question:
1. Provide the question.
2. Give a **detailed answer** with a full explanation.
3. Include **code snippets, best practices, and troubleshooting techniques** where necessary.
4. Ensure **clear, structured, and in-depth responses** without any word limits.

Provide all 30 questions with their corresponding answers and explanations.
"""

if submit_resume:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt_resume, st.session_state.session_id)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit_match:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt_match, st.session_state.session_id)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit_learning:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt_learning, st.session_state.session_id)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit_enhance:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt_enhance, st.session_state.session_id)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

# Handling Interview Questions with Answers and PDF Download
elif submit_python:
    response = get_gemini_response_question(input_prompt_python, st.session_state.session_id)
    st.subheader("Python Interview Questions & Answers:")
    st.write(response)
    
    pdf_buffer = generate_pdf(response)
    st.download_button(label="Download PDF", 
                       data=pdf_buffer.getvalue(),  # Convert to bytes
                       file_name="python_questions.pdf", 
                       mime="application/pdf")
elif submit_ml:
    response = get_gemini_response_question(input_prompt_ml, st.session_state.session_id)
    st.subheader("Machine Learning Interview Questions & Answers:")
    st.write(response)
    
    pdf_buffer = generate_pdf(response)
    st.download_button(label="Download PDF", 
                       data=pdf_buffer.getvalue(),  # Convert to bytes
                       file_name="ML.pdf", 
                       mime="application/pdf")
elif submit_dl:
    response = get_gemini_response_question(input_prompt_dl, st.session_state.session_id)
    st.subheader("Deep Learning Interview Questions & Answers:")
    st.write(response)
    
    pdf_buffer = generate_pdf(response)
    st.download_button(label="Download PDF", 
                       data=pdf_buffer.getvalue(),  # Convert to bytes
                       file_name="DL.pdf", 
                       mime="application/pdf")
elif submit_docker:
    response = get_gemini_response_question(input_prompt_docker, st.session_state.session_id)
    st.subheader("Docker Interview Questions & Answers:")
    st.write(response)
    
    pdf_buffer = generate_pdf(response)
    st.download_button(label="Download PDF", 
                       data=pdf_buffer.getvalue(),  # Convert to bytes
                       file_name="Docker.pdf", 
                       mime="application/pdf")

# Text Input


    

