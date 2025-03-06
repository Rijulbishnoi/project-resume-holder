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