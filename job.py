import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("JSEARCH_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Data Science & Analytics Jobs", layout="wide")
st.header("Explore Data Science & Analytics Jobs")
st.subheader("Click on a company to view job description:")

# List of companies
companies = ["TCS", "Wipro", "Infosys", "Accenture", "Cognizant"]

# Function to fetch jobs using JSearch API
def fetch_jobs(company):
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"{company} Data Scientist", "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": API_KEY,
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
            st.write(f"*Company:* {job.get('employer_name', 'N/A')}")
            st.write(f"*Location:* {job.get('job_city', 'Unknown')}, {job.get('job_country', 'Unknown')}")
            st.write(f"*Description:* {job.get('job_description', 'No description available.')}")
            st.markdown(f"[Apply Here]({job.get('job_apply_link', '#')})")
            st.write("---")
    else:
        st.write("No job listings found. Try again later!")