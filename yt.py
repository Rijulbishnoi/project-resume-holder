import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Function to fetch YouTube video transcript
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
st.set_page_config(page_title="YouTube Video Analyzer", layout="wide")
    
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