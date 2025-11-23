import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# pyKT imports
from pykt.models import DKT
from pykt.preprocessing import ResponseSequence, KCMapping

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="AI Study Mentor (Secure Groq)", page_icon="📘", layout="wide")

# ------------------------------
# Load Groq API key from Streamlit Secrets
# ------------------------------
try:
    GROQ_API_KEY = st.secrets["groq"]["API_KEY"]
    GROQ_MODEL = st.secrets["groq"]["MODEL"]
except KeyError:
    GROQ_API_KEY = None
    GROQ_MODEL = "llama-3.3-70b-versatile"
    st.warning("Groq API key not found in Streamlit Secrets. Falling back to local model.")

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    gen_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name)
    gen_pipe = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer)

    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    return gen_pipe, sentiment, embed

gen_pipe, sentiment_pipe, embed_model = load_models()

# ------------------------------
# Initialize Knowledge Tracing
# ------------------------------
@st.cache_resource
def init_kt_model(num_kcs):
    model = DKT(n_items=num_kcs, hidden_size=32, n_layers=1)
    return model

# ------------------------------
# Groq API call
# ------------------------------
def call_groq(prompt, max_tokens=700):
    if not GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"Groq API call failed: {e}")
        return None

# ------------------------------
# Text Generation
# ------------------------------
def generate_text(prompt):
    api_out = call_groq(prompt)
    if api_out:
        return api_out
    out = gen_pipe(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"]

# ------------------------------
# OCR Extraction
# ------------------------------
def ocr_extract(file, dpi=150, max_pages=50):
    name = file.name.lower()
    if name.endswith(".pdf"):
        data = file.read()
        images = convert_from_bytes(data, dpi=dpi)[:max_pages]
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text
    else:
        img = Image.open(file)
        return pytesseract.image_to_string(img)

# ------------------------------
# Chunk Text
# ------------------------------
def chunk_text(text, max_chars=2000):
    words = text.split()
    chunks = []
    cur = []
    length = 0
    for w in words:
        cur.append(w)
        length += len(w) + 1
        if length >= max_chars:
            chunks.append(" ".join(cur))
            cur = []
            length = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ------------------------------
# UI
# ------------------------------
st.title("📘 AI Study Mentor — Secure with Streamlit Secrets")

uploaded = st.file_uploader("Upload PDF / Image for study", type=["pdf","jpg","png","jpeg"])
max_pages = st.sidebar.slider("Max pages to OCR", 5, 200, 50)

if uploaded:
    with st.spinner("Extracting text..."):
        full_text = ocr_extract(uploaded, max_pages=max_pages)
    if not full_text.strip():
        st.error("No text extracted.")
    else:
        chunks = chunk_text(full_text)
        st.success(f"Text split into {len(chunks)} chunks.")

        # Generate Notes, MCQs, Flowcharts
        notes = [generate_text(f"Summarize into study notes:\n\n{ch}") for ch in chunks]
        mcqs = [generate_text(f"Generate 5 MCQs with options + answer:\n\n{ch}") for ch in chunks]
        flows = [generate_text(f"Convert into Mermaid flowchart:\n\n{ch}") for ch in chunks]

        st.header("📝 Notes")
        for i, n in enumerate(notes):
            st.subheader(f"Chunk {i+1}")
            st.write(n)

        st.header("❓ MCQs")
        for i, m in enumerate(mcqs):
            st.subheader(f"Chunk {i+1}")
            st.write(m)

        st.header("📊 Flowcharts")
        for i, f in enumerate(flows):
            st.subheader(f"Chunk {i+1}")
            st.markdown("```mermaid\n" + f + "\n```")

        # Embeddings
        embs = embed_model.encode(chunks, convert_to_numpy=True)
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embs

        # Initialize KT model
        if "kt_model" not in st.session_state:
            st.session_state["kt_model"] = init_kt_model(len(chunks))
            st.session_state["seq"] = ResponseSequence()

        # Practice MCQs
        st.header("🧪 Practice MCQs")
        chunk_index = st.number_input("Select chunk", 1, len(chunks), 1) - 1
        if mcqs[chunk_index]:
            st.write(mcqs[chunk_index])
            user_ans = st.text_input("Type your answer (e.g. A, B, C, D)")
            if st.button("Submit your answer"):
                correct = st.radio("Was your answer correct?", ("Yes", "No"))
                obs = 1 if correct == "Yes" else 0
                kc_map = KCMapping({i: i for i in range(len(chunks))})
                st.session_state["seq"].add_response(kc_map, chunk_index, obs)
                st.session_state["kt_model"].fit(st.session_state["seq"])
                mastery = st.session_state["kt_model"].predict(st.session_state["seq"])
                st.session_state["mastery"] = mastery[0].tolist()
                st.success(f"Updated mastery: {st.session_state['mastery']}")

        # Recommend weak topics
        if "mastery" in st.session_state:
            st.header("🔍 Topics to Review")
            probs = st.session_state["mastery"]
            weakest = sorted(range(len(probs)), key=lambda i: probs[i])[:3]
            for w in weakest:
                st.write(f"Chunk {w+1}: mastery ≈ {probs[w]:.2f}")

        # Q&A with RAG + sentiment
        st.header("❓ Ask Mentor a Question")
        question = st.text_area("Your question:", height=150)
        if st.button("Get Answer"):
            if "embeddings" in st.session_state:
                q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
                sims = (st.session_state["embeddings"] @ q_emb) / (
                    np.linalg.norm(st.session_state["embeddings"], axis=1) * np.linalg.norm(q_emb)
                )
                top_idx = sims.argsort()[-3:][::-1]
                context = "\n\n".join([st.session_state["chunks"][i] for i in top_idx])

                sentiment = sentiment_pipe(question[:512])[0]["label"]
                style = "supportive, step-by-step" if sentiment == "NEGATIVE" else "clear and detailed"

                prompt = f"You are a professional tutor. Style: {style}. Context: {context}\n\nQuestion: {question}"
                answer = generate_text(prompt)

                prompt_flow = f"Explain the answer as a Mermaid flowchart. Context: {context}\n\nQuestion: {question}"
                flowchart = generate_text(prompt_flow)

                st.subheader("🧠 Answer")
                st.write(answer)
                st.subheader("📊 Flowchart Explanation")
                st.markdown("```mermaid\n" + flowchart + "\n```")
            else:
                st.error("Upload material first to initialize knowledge base.")
