from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
from fpdf import FPDF
import requests  # Needed for downloading the font file

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

def ensure_font_exists(font_path):
    """Check if the font file exists; if not, download it."""
    if not os.path.exists(font_path):
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
        st.info(f"Downloading DejaVuSans.ttf from {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(font_path, "wb") as f:
                f.write(response.content)
            st.success(f"Downloaded DejaVuSans.ttf to {font_path}")
        else:
            raise FileNotFoundError(f"Font file not found at {font_path} and download failed with status code {response.status_code}.")

def get_gemini_response(input_text, pdf_content, prompt):
    """Generate a response using Google Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    """Convert first page of uploaded PDF to an image and encode as base64."""
    if uploaded_file is not None:
        uploaded_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(uploaded_file.read())
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

def generate_pdf(updated_resume_text):
    """Generate a downloadable PDF file with Unicode support."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Define correct font path and ensure it exists
    import os
    font_path = os.path.join(os.getcwd(), "fonts/DejaVuSans.ttf")
    
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    # Wrap long text in a multi-cell that fits A4 width
    pdf.multi_cell(190, 10, updated_resume_text, align="L")

    pdf_output_path = "updated_resume.pdf"
    pdf.output(pdf_output_path, "F")
    return pdf_output_path

#function for independently generate output
def get_gemini_response_question(prompt):
    """Generate a response using Google Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt])
    return response.text

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("MY A5 PERSONAL ATS")

st.subheader("Resume analysis")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

if uploaded_file:
    st.success("PDF Uploaded Successfully.")
    pdf_content = input_pdf_setup(uploaded_file)

# Define buttons
col0,col1, col2, col3, col4 = st.columns(5)
with col0:
    submit_tell_me_about=st.button("Tell Me About Resume")
with col1:
    submit_match = st.button("Percentage Match")
with col2:
    submit_learning = st.button("Personalized Learning Path")
with col3:
    submit_update = st.button("Update Resume & Download")
with col4:
   submit_interview = st.button("Generate Interview Questions")

input_prompts = {
    "Tell_me_about_resume":"""
    You are an expert resume writer with deep knowledge of Data Science, Full Stack, Web Development, 
    Big Data Engineering, DevOps, and Data Analysis. Your task is to refine and optimize the provided resume 
    according to the job description. Ensure the new resume:
    - Highlights relevant experience and skills.
    - Optimizes for ATS (Applicant Tracking Systems).
    - Uses strong action words and quantifiable achievements.
    - Incorporates key industry keywords.
    """,

    "percentage_match": """
    You are an ATS evaluator. Provide:
    1. An overall match percentage.
    2. Breakdown: 
    - Skills match (% weight)
    - Experience match (% weight)
    - Keyword relevance (% weight)
    3. What can be improved to increase the match?

    """,

    "personalized_learning": """
    You are an experienced learning coach and technical expert. Create a 6-month personalized study plan 
    for an individual aiming to excel in [Job Role], focusing on the skills, topics, and tools specified 
    in the provided job description. Ensure the study plan includes:
    - A list of topics and tools for each month.
    - Suggested resources (books, online courses, documentation).
    - Recommended practical exercises or projects.
    - Periodic assessments or milestones.
    - Tips for real-world applications.
    """,

    "resume_update": """
    You are an expert resume writer with deep knowledge of Data Science, Full Stack, Web Development, 
    Big Data Engineering, DevOps, and Data Analysis. Your task is to refine and optimize the provided resume 
    according to the job description. Ensure the new resume:
    - Highlights relevant experience and skills.
    - Optimizes for ATS (Applicant Tracking Systems).
    - Uses strong action words and quantifiable achievements.
    - Incorporates key industry keywords.
    """,

    "interview_questions": """
    You are an AI-powered interview coach.
    Generate {num_questions} interview questions based on the given job description, 
    focusing on the required skills and expertise.
    """,

    "question_bank": """
    Generate {num_questions} {level}-level interview questions on {topic} with answers.
    """
}

if submit_tell_me_about and uploaded_file:
    response=get_gemini_response(input_text,pdf_content,input_prompts["Tell_me_about_resume"])
    st.subheader("Tell_me_about_resume:")
    st.write(response)
    
elif submit_match and uploaded_file:
    response = get_gemini_response(input_text, pdf_content, input_prompts["percentage_match"])
    st.subheader("Percentage Match:")
    st.write(response)

elif submit_learning and uploaded_file:
    response = get_gemini_response(input_text, pdf_content, input_prompts["personalized_learning"])
    st.subheader("Personalized Learning Path:")
    st.write(response)

elif submit_update and uploaded_file:
    response = get_gemini_response(input_text, pdf_content, input_prompts["resume_update"])
    if response:
        pdf_path = generate_pdf(response)
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Updated_Resume.pdf">Download Updated Resume</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Error generating updated resume.")
    

#DATA SCIENCE QUESTION BANK
# Streamlit App - Data Science Question Bank with Select Box
st.subheader("Interview Question Bank")

if "show_question_selection" not in st.session_state:
    st.session_state.show_question_selection = False# Initialize state

if st.button("Question And Answer"):
    st.session_state.show_question_selection = True  # Set to True when clicked


if st.session_state.show_question_selection:
    
    fields = st.multiselect("Select a field:", ["Data Analyst", "Data Engineer", "Data Scientist"])
    topic_mapping = {
        "Data Scientist": ["Core Python", "Machine Learning", "Deep Learning", "Generative AI"],
        "Data Analyst": ["SQL", "Data Visualization", "Statistics", "Python for Data Analysis"],
        "Data Engineer": ["Big Data", "ETL Pipelines", "Cloud & DevOps", "Database Management"]
    }

    selected_topics = []
    for field in fields:
        selected_topics += st.multiselect(f"Select topics for {field}:", topic_mapping[field])

    level = st.selectbox("Select Difficulty Level:", ["Easy", "Intermediate", "Difficult"])
    num_questions = st.slider("Number of questions per topic:", 1, 20, 5)

    generate_button = st.button("Generate")

    if generate_button and fields and selected_topics:
        all_questions = ""
        
        for topic in selected_topics:
            prompt = f"Generate {num_questions} {level}-level interview questions on {topic} with answers."
            response = get_gemini_response_question(prompt)
            all_questions += f"\n\n### {topic} ({level})\n" + response
        
        st.subheader("Generated Questions:")
        st.write(all_questions) 
        
## top 5 MNcs
# Define buttons of MNCs

st.subheader("Top 5 MNCs")

selected_company = st.selectbox("Select a company:", ["Amazon", "Google", "Meta", "IBM", "Nvidia"])

if uploaded_file and selected_company:
    pdf_content = input_pdf_setup(uploaded_file)  # Extract resume content

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        project_btn = st.button("Projects Required", key=f"{selected_company}_projects")

    with col2:
        skills_btn = st.button("Skills Required", key=f"{selected_company}_skills")

    with col3:
        recommend_btn = st.button("Recommendations", key=f"{selected_company}_recommend")

    with col4:
        match_btn = st.button("Resume Match Score", key=f"{selected_company}_match")

    prompt_company = {
        "Projects": f"""
        You are a Data Science career advisor specializing in {selected_company}. 
        Analyze the provided resume and job description. Suggest *real-world projects* 
        that would strengthen the candidate’s profile for a Data Science role at {selected_company}.
        """,
        
        "Skills": f"""
        You are a recruiter at {selected_company}. Based on the provided resume, 
        analyze missing *technical and soft skills* required for a Data Science position at {selected_company}. 
        Highlight skills the candidate already has and suggest improvements.
        """,
        
        "Recommendations": f"""
        You are an expert in hiring Data Scientists at {selected_company}. 
        Based on the resume, provide *personalized recommendations* on:
        - How to tailor the resume to align better with {selected_company}.
        - How to optimize the LinkedIn profile for visibility.
        - Additional resources (books, courses, projects) to improve chances of selection.
        """,

        "MatchScore": f"""
        You are an ATS (Applicant Tracking System) evaluator for {selected_company}.  
        Compare the provided *resume* with a standard *Data Scientist job description at {selected_company}*.  
        Provide a *match score (out of 100)* based on:
        - Relevant experience  
        - Required technical skills (Python, SQL, ML, etc.)  
        - Soft skills (communication, teamwork, etc.)  
        - Projects and past work  

        Also, suggest *specific improvements* to increase the match percentage.
        """
    }

    if project_btn:
        response = get_gemini_response(input_text, pdf_content, prompt_company["Projects"])
        st.subheader(f"{selected_company} - Projects Required")
        st.write(response)

    elif skills_btn:
        response = get_gemini_response(input_text, pdf_content, prompt_company["Skills"])
        st.subheader(f"{selected_company} - Skills Required")
        st.write(response)

    elif recommend_btn:
        response = get_gemini_response(input_text, pdf_content, prompt_company["Recommendations"])
        st.subheader(f"{selected_company} - Recommendations")
        st.write(response)

    elif match_btn:
        response = get_gemini_response(input_text, pdf_content, prompt_company["MatchScore"])
        st.subheader(f"{selected_company} - Resume Match Score")
        st.write(response)
else:
    st.warning("Please upload your resume to get personalized insights.")




## DSA for Data science
st.subheader("DSA For Data Science")

dsa_level=st.selectbox("Select Difficulty Level:",["Easy","Medium","Hard"])
num_questions=10
prompt_dsa = f"""
Generate {num_questions} {dsa_level}-level Data Structures and Algorithms (DSA) questions 
specifically relevant for Data Science roles. Focus on concepts such as:
- Arrays, Linked Lists, and Hash Maps
- Searching and Sorting (QuickSort, MergeSort, Binary Search)
- Dynamic Programming (Knapsack, LCS, etc.)
- Graph Algorithms (BFS, DFS, Dijkstra's, PageRank)
- Trees and Tries (Binary Search Trees, Heaps)
- String Manipulation and Pattern Matching (KMP, Rabin-Karp)
- Time Complexity and Optimization Techniques

For each question, provide:
1. A clear problem statement
2. Constraints and example test cases
3. A detailed explanation of the optimal approach
4. Python code implementation
"""

if st.button(f"Generate {dsa_level} Questions"):
    response = get_gemini_response_question(prompt_dsa)
    st.subheader(f"{dsa_level}-Level DSA Questions for Data Science")
    st.write(response)

dsa_topic=st.selectbox("Select DSA Topic:",["Array","Recursion","Linkedlist","Queue","Tree","Graphs","Dynamic Programming"])
dsa_topic_prompt={"""Teach me {dsa_topic} with case studies.

Explain {dsa_topic} in detail, covering its concept, importance, and real-world applications. Provide:

Concept Explanation – Define {dsa_topic} and explain its significance in programming and problem-solving.
Case Studies – Provide at least two real-world case studies demonstrating the use of {dsa_topic} in domains like Data Science, Machine Learning, Web Development, or System Design.
Problem-Solving Approach – Explain how {dsa_topic} helps solve practical problems and compare it with alternative methods.
Code Implementation – Provide a Python implementation, with step-by-step explanations and test cases.
Best Practices & Optimizations – Discuss common pitfalls, best practices, and performance optimizations.
Keep the explanation structured, beginner-friendly, and engaging, with clear examples and insights for experienced programmers."""}

if st.button(f"Teach me  {dsa_topic} with case studies "):
    response = get_gemini_response_question(prompt_dsa)
    st.subheader(f"{dsa_level}-Level DSA Questions for Data Science")
    st.write(response)

import streamlit as st
import librosa
import numpy as np
import tempfile
from pydub import AudioSegment
import os
import ffmpeg

# Ensure ffmpeg is installed
FFMPEG_PATH = "C:\\Users\\Abc\\Downloads\\ffmpeg-7.1-full_build\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe"
FFPROBE_PATH = "C:\\Users\\Abc\\Downloads\\ffmpeg-7.1-full_build\\ffmpeg-7.1-full_build\\bin\\ffprobe.exe"

st.subheader("🎤 Pitch & Confidence Analyzer from Video")

uploaded_video = st.file_uploader("Upload your video (MP4, AVI, MOV)...", type=['mp4', 'avi', 'mov'])

if uploaded_video:
    st.success("✅ Video Uploaded Successfully.")

    # Save uploaded video temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video_path = temp_video.name

    # Extract audio from video using pydub
    audio_path = temp_video_path.replace(".mp4", ".wav")

    try:
        audio = AudioSegment.from_file(temp_video_path, format="mp4")  
        audio.export(audio_path, format="wav")
    except Exception as e:
        st.error(f"❌ Error extracting audio: {e}")
        st.stop()

    # Load the extracted audio for analysis
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        st.error(f"❌ Error loading audio: {e}")
        st.stop()

    # 🔹 Debugging: Check if y has data
    if len(y) == 0:
        st.error("❌ No valid audio data found. Please upload a different video.")
        st.stop()

    # 🔹 Improved Pitch Extraction using YIN algorithm
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400)  # Adjusted fmin & fmax to capture human voice
        valid_f0 = f0[f0 > 0]  # Remove zero values
    except Exception as e:
        st.error(f"❌ Error in pitch tracking: {e}")
        st.stop()

    # 🔹 Debugging: Check extracted pitch values
    st.write(f"*Debug: Extracted {len(valid_f0)} pitch values*")
    if len(valid_f0) == 0:
        st.error("❌ No valid pitch detected. Try another video with clearer speech.")
        st.stop()

    avg_pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 0
    pitch_variability = np.std(valid_f0) if len(valid_f0) > 0 else 0

    # 🔹 Fix Confidence Score Calculation
    confidence_score = max(0, min(100, 100 - (pitch_variability * 1.5)))  # Less strict variation penalty

    # 🧐 Define confidence rating
    if confidence_score > 75:
        confidence_label = "High 🎯"
        confidence_color = "✅"
    elif confidence_score > 50:
        confidence_label = "Moderate ⚠"
        confidence_color = "🟡"
    else:
        confidence_label = "Low ❌"
        confidence_color = "🔴"

    # 🎙 Display analysis results
    st.write("### 🎙 Speech Analysis Results")
    st.write(f"*Average Pitch:* {avg_pitch:.2f} Hz 🎵")
    st.write(f"*Pitch Variability:* {pitch_variability:.2f}")
    st.write(f"*Confidence Score:* {confidence_score:.1f}/100 {confidence_color} ({confidence_label})")

    # 🧐 Insights & Recommendations
    st.subheader("📊 Insights & Recommendations")
    if confidence_score > 75:
        st.success("✅ Your pitch and voice control indicate strong confidence! Keep up the good work.")
    elif confidence_score > 50:
        st.warning("⚠ Your confidence is moderate. Try maintaining a steady voice to improve clarity.")
    else:
        st.error("❌ Your speech shows low confidence. Practice speaking with a stable tone and controlled pitch.")

    # Optional: Play extracted audio
    st.audio(audio_path, format="audio/wav")

    # Cleanup: Remove temporary files
    os.remove(temp_video_path)
    os.remove(audio_path)