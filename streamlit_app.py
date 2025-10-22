import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import pdfplumber

# ---------------- Page Setup ----------------
st.set_page_config(page_title="📊 Automated Report Summarizer", page_icon="🧠")
st.title("📊 Automated Multi-Input Report Summarizer ")
st.write("Upload TXT, PDF, DOCX, CSV, Audio, or Chart Image to generate a detailed summary.")

# ---------------- Gemini API Key ----------------
api_key = "AIzaSyBEQTNvHmh5hvqHspCCW1-ri2f50aa_tDY"  # 🔑 Replace with your real Gemini API key
genai.configure(api_key=api_key)

# ---------------- Helper Functions ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    from docx import Document
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_csv(file):
    import pandas as pd
    df = pd.read_csv(file)
    return "Here is a summary of the uploaded data:\n" + df.to_string(index=False)

def transcribe_audio(file):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# ---------------- Gemini Text Summarization ----------------
def summarize_with_gemini(text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([
            {
                "parts": [
                    {"text": f"Please provide a detailed, well-structured summary of the following text:\n{text}"}
                ]
            }
        ])
        return response.text
    except Exception as e:
        return f"⚠️ summarization failed: {e}"

# ---------------- Gemini Image Analysis ----------------
def analyze_chart_with_gemini(image_path):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = model.generate_content([
            {
                "parts": [
                    {"text": "Analyze this chart or plot image and summarize key trends, insights, and business implications."},
                    {"mime_type": "image/png", "data": img_bytes}
                ]
            }
        ])
        return response.text
    except Exception as e:
        return f"⚠️ analysis failed: {e}"

# ---------------- File Uploader ----------------
uploaded_file = st.file_uploader(
    "📂 Upload report, audio, or chart image",
    type=["txt", "pdf", "docx", "csv", "wav", "mp3", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = None

    # ---------------- Image ----------------
    if file_ext in [".png", ".jpg", ".jpeg"]:
        try:
            image = Image.open(uploaded_file)
            try:
                st.image(uploaded_file, caption="Uploaded Chart/Image", use_container_width=True)
            except TypeError:
                st.image(uploaded_file, caption="Uploaded Chart/Image", use_column_width=True)

            st.info("🧠 Analyzing image ...")
            temp_path = "temp_chart.png"
            image.save(temp_path)
            chart_summary = analyze_chart_with_gemini(temp_path)

            st.subheader("🖼️  Image Summary")
            st.success(chart_summary)

            text = f"AI Summary of chart:\n{chart_summary}"

        except Exception as e:
            st.error(f"Error processing image: {e}")

    # ---------------- Text / PDF / DOCX / CSV / Audio ----------------
    else:
        try:
            if file_ext == ".txt":
                text = uploaded_file.read().decode("utf-8")
            elif file_ext == ".pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif file_ext == ".docx":
                text = extract_text_from_docx(uploaded_file)
            elif file_ext == ".csv":
                text = extract_text_from_csv(uploaded_file)
            elif file_ext in [".wav", ".mp3"]:
                st.info("🎙️ Transcribing audio... please wait")
                text = transcribe_audio(uploaded_file)
            else:
                st.error("Unsupported file type!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # ---------------- Summarization ----------------
    if text:
        st.subheader("📄 Extracted Text / Insights")
        st.write(text[:1000] + "..." if len(text) > 1000 else text)

        with st.spinner("🧩 Generating detailed summary ..."):
            chunks = list(chunk_text(text, max_words=500))
            detailed_summary = ""
            for chunk in chunks:
                chunk_summary = summarize_with_gemini(chunk)
                detailed_summary += chunk_summary + "\n\n"

        st.subheader("📝 Final Generated Summary")
        st.success(detailed_summary)

        # Save output
        os.makedirs("output", exist_ok=True)
        output_path = "output/summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(detailed_summary)

        st.info(f"📁 Summary saved to `{output_path}`")
        st.download_button(
            label="⬇️ Download Summary",
            data=detailed_summary,
            file_name="summary.txt",
            mime="text/plain"
        )

else:
    st.info("Please upload a file or chart image to begin summarization.")
