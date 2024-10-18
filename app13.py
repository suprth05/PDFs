import os
import re
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from gtts import gTTS

# Load Hugging Face API key
def load_huggingface_api_key():
    dotenv_path = "huggingface.env"
    load_dotenv(dotenv_path)
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        raise ValueError(f"Unable to retrieve HUGGINGFACE_API_KEY from {dotenv_path}")
    return hf_api_key

# Preprocessing: Clean the extracted text by removing unwanted patterns
def clean_text(text):
    text = re.sub(r"R V College of Engineering", "", text)
    text = re.sub(r"Computer Science Stream.*", "", text)
    text = re.sub(r"\d+", "", text)  # Remove any stray numbers or page numbers
    text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
    return text

# Split the text into coherent chunks based on specific topics or structure
def split_into_chunks(text):
    sections = re.split(r'(Introduction|Assumptions|Drift Velocity|Band theory)', text)
    chunks = []
    for i in range(1, len(sections), 2):
        section_title = sections[i]
        section_content = sections[i + 1]
        chunk = section_title + " " + section_content
        chunks.append(chunk.strip())
    return chunks

# Summarize each chunk by calling the Hugging Face API
def summarize_chunk_hf(chunk, hf_api_key, max_len, min_len):
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    data = {"inputs": chunk, "parameters": {"max_length": max_len, "min_length": min_len, "do_sample": False}}
    
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return summary
    else:
        raise Exception(f"Failed to get a response from Hugging Face API: {response.status_code}, {response.text}")

# Summarize all chunks by calling the Hugging Face API for each one
def summarize_chunks(chunks, hf_api_key, max_len, min_len):
    summaries = []
    for chunk in chunks:
        summary = summarize_chunk_hf(chunk, hf_api_key, max_len, min_len)
        summaries.append(summary)
    return summaries

# Ensure the audio directory exists
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Text-to-Speech function with directory storage
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file_name = os.path.join(AUDIO_DIR, "summary.mp3")  # Save audio as summary.mp3
    tts.save(audio_file_name)
    return audio_file_name  # Return the name of the audio file

# Main app
def main():
    st.set_page_config(page_title="PDF Summarizer & Text-to-Speech", page_icon="📄", layout="wide")
    
    # Add a sidebar with some information
    st.sidebar.image("https://image.shutterstock.com/image-vector/document-summary-icon-260nw-2019121439.jpg", use_column_width=True)
    st.sidebar.markdown("## PDF Summarizer")
    st.sidebar.markdown("Upload a PDF file, summarize it, and listen to the summarized audio!")

    st.title("📄 PDF Summarizer with Voice Explanation")
    st.write("Created by Team Troupe")

    # Split layout into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload the PDF document
        pdf = st.file_uploader('📂 **Upload Your PDF Document Here:**', type='pdf')
    
    with col2:
        # Customizable summary length
        max_len = st.slider('🔍 Max Length of Summary:', 100, 500, 300)
        min_len = st.slider('🔍 Min Length of Summary:', 50, 150, 80)

    if pdf is not None:
        # Add a progress bar
        progress = st.progress(0)
        
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Clean the extracted text
        cleaned_text = clean_text(text)

        # Split the cleaned text into chunks
        chunks = split_into_chunks(cleaned_text)

        # Summarize each chunk and combine results
        summaries = summarize_chunks(chunks, hf_api_key, max_len, min_len)
        full_summary = "\n\n".join(summaries)  # Organize with newlines between sections
        
        # Update progress bar
        progress.progress(100)
        
        # Display the final summarized output with structured sections
        st.subheader('📝 Summary Results:')
        st.markdown(f"<div style='background-color: #F0F2F6; padding: 10px;'>{full_summary}</div>", unsafe_allow_html=True)

        # Add Text-to-Speech functionality
        if st.button('🔊 Listen to Summary'):
            audio_file = text_to_speech(full_summary)  # Save to file
            st.audio(audio_file, format='audio/mp3')  # Play the audio
            st.success("Audio generated successfully! 🎧")
    
if __name__ == '__main__':
    main()

