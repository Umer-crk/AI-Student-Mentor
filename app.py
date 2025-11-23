import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pdf2image import convert_from_bytes
import requests
from io import BytesIO

# -------------------------------
# Groq API settings (use Streamlit secrets)
# -------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

def call_groq(prompt, max_tokens=700):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"API Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Request Error: {e}"

# -------------------------------
# OCR
# -------------------------------
def ocr_extract(file, dpi=150):
    data = file.read()
    try:
        pages = convert_from_bytes(data, dpi=dpi)
    except Exception as e:
        st.error("Error processing PDF. Ensure Poppler is installed.")
        return ""
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text

# -------------------------------
# Chunking text
# -------------------------------
def split_text_chunks(text, max_chars=3000):
    words = text.split()
    chunks, chunk = [], ""
    for w in words:
        if len(chunk)+len(w)+1 <= max_chars:
            chunk += " " + w
        else:
            chunks.append(chunk.strip())
            chunk = w
    if chunk.strip(): chunks.append(chunk.strip())
    return chunks

def call_groq_chunks(text, prompt_fn):
    results = []
    for chunk in split_text_chunks(text):
        results.append(call_groq(prompt_fn(chunk)))
    return "\n\n".join(results)

# -------------------------------
# Prompts
# -------------------------------
def prompt_mcqs(txt):
    return f"Generate 10 professional MCQs with 4 options and answers from the text:\n\n{txt}"

def prompt_notes(txt):
    return f"Summarize the text into professional study notes with bullets and headings:\n\n{txt}"

def prompt_flowchart(txt):
    return f"Convert content into a step-by-step flowchart with arrows ->:\n\n{txt}"

# -------------------------------
# Flowchart image
# -------------------------------
def make_flowchart_image(text):
    W,H = 1200,900
    img = Image.new("RGB",(W,H),(255,255,255))
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf",16)
    except: font = ImageFont.load_default()
    y = 50
    for line in text.split("\n")[:25]:
        draw.text((50,y), line, font=font, fill="black")
        y += 35
    return img

# -------------------------------
# Sentiment-aware question answering
# -------------------------------
def answer_question(q):
    if not q.strip(): return "Enter a question."
    sentiment_prompt = f"Detect sentiment (POSITIVE, NEGATIVE, NEUTRAL) of this question:\n{q}"
    sentiment = call_groq(sentiment_prompt, max_tokens=50).strip().upper()
    style_map = {
        "NEGATIVE":"Step-by-step, simple, explanatory.",
        "POSITIVE":"Detailed, professional, insightful.",
        "NEUTRAL":"Clear, structured explanation."
    }
    style = style_map.get(sentiment,"Clear, structured explanation.")
    answer_prompt = f"Answer the question professionally in this style: {style}\nQuestion: {q}"
    answer = call_groq(answer_prompt, max_tokens=500)
    flow_img = make_flowchart_image(answer)
    return f"Sentiment: {sentiment}\nAnswer:\n{answer}", flow_img

# -------------------------------
# Streamlit Interface
# -------------------------------
st.set_page_config(page_title="📚 AI Student Mentor", layout="wide")
st.title("📚 AI Student Mentor (Free PDF Limit: 2MB/day)")

tabs = st.tabs(["Upload PDF", "Ask a Question"])

# -------------------------------
# Tab 1: Upload PDF
# -------------------------------
with tabs[0]:
    uploaded = st.file_uploader(
        label="📄 Upload your PDF (Limit 2MB per file. PDF)",  # Fixed label
        type=["pdf"]
    )
    
    if uploaded:
        max_size = 2 * 1024 * 1024  # 2MB in bytes
        if uploaded.size > max_size:
            st.error(f"❌ PDF size exceeds 2MB. Your file size: {uploaded.size/1024/1024:.2f} MB")
        else:
            st.info("⏳ Processing PDF...")
            full_text = ocr_extract(uploaded)
            if full_text.strip():
                mcqs = call_groq_chunks(full_text, prompt_mcqs)
                notes = call_groq_chunks(full_text, prompt_notes)
                flowchart_text = call_groq_chunks(full_text, prompt_flowchart)
                flow_img = make_flowchart_image(flowchart_text)
                
                st.subheader("✅ Generated MCQs & Notes")
                st.text_area("MCQs & Notes", mcqs + "\n\n" + notes, height=300)
                
                st.subheader("📊 Flowchart")
                st.image(flow_img)

# -------------------------------
# Tab 2: Ask a Question
# -------------------------------
with tabs[1]:
    q_in = st.text_area("Enter your question here:", "", height=100)
    if st.button("Get Answer"):
        if q_in.strip():
            with st.spinner("AI is generating answer and flowchart..."):
                ans_text, ans_img = answer_question(q_in)
                st.subheader("📝 Answer")
                st.text_area("", ans_text, height=200)
                st.subheader("📊 Flowchart Explanation")
                st.image(ans_img)
