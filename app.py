import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config & Secrets
# -----------------------------
st.set_page_config(page_title="AI Study Mentor (Stable)", page_icon="📘", layout="wide")

# Load Groq API key from secrets
try:
    GROQ_API_KEY = st.secrets["groq"]["API_KEY"]
    GROQ_MODEL = st.secrets["groq"]["MODEL"]
except Exception:
    GROQ_API_KEY = None
    GROQ_MODEL = "llama‑3.3‑70b‑versatile"
    st.warning("Groq API key not found in Streamlit Secrets — using local model fallback.")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    gen_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name)
    gen_pipe = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer)

    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    return gen_pipe, sentiment_pipe, embed_model

gen_pipe, sentiment_pipe, embed_model = load_models()

# -----------------------------
# Utility functions
# -----------------------------
def call_groq(prompt, max_tokens=700):
    if not GROQ_API_KEY:
        return None
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role":"user","content":prompt}],
                "max_tokens": max_tokens
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"Groq API call failed: {e}")
        return None

def generate_text(prompt):
    out = call_groq(prompt)
    if out:
        return out
    result = gen_pipe(prompt, max_length=512, do_sample=False)
    return result[0]["generated_text"]

def ocr_extract(file, dpi=150, max_pages=50):
    name = file.name.lower()
    if name.endswith(".pdf"):
        data = file.read()
        pages = convert_from_bytes(data, dpi=dpi)[:max_pages]
        text = "\n".join([pytesseract.image_to_string(page) for page in pages])
        return text
    else:
        img = Image.open(file)
        return pytesseract.image_to_string(img)

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

# -----------------------------
# UI
# -----------------------------
st.title("📘 AI Study Mentor — Free/Stable Edition")
st.markdown("""
This app uses:
- Real‑time PDF/Image upload → Notes, MCQs, Flowcharts  
- Retrieval‑augmented generation (RAG) via embeddings  
- Sentiment‑aware tutoring style  
- Local models + optional Groq API  
""")

# Remind about secrets
if GROQ_API_KEY is None:
    st.info("To use the remote generation (Groq), please set the API key in `.streamlit/secrets.toml` under [groq] section.")

uploaded = st.file_uploader("Upload PDF or Image (recommended ≤ 2MB for free use)", type=["pdf","png","jpg","jpeg"])

max_pages = st.sidebar.slider("Max pages to OCR", 5, 200, 50)

if uploaded:
    with st.spinner("Extracting text from file..."):
        full_text = ocr_extract(uploaded, max_pages=max_pages)
    if not full_text.strip():
        st.error("No extractable text found.")
    else:
        chunks = chunk_text(full_text)
        st.success(f"Split into {len(chunks)} chunks.")

        # Generate study materials
        notes = [ generate_text(f"Summarize into study notes:\n\n{ch}") for ch in chunks ]
        mcqs  = [ generate_text(f"Generate 5 MCQs (4 options + answer) from:\n\n{ch}") for ch in chunks ]
        flows = [ generate_text(f"Convert into Mermaid flowchart:\n\n{ch}") for ch in chunks ]

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

        # Build embeddings for retrieval
        embs = embed_model.encode(chunks, convert_to_numpy=True)
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embs

        # Simple mastery tracking using counts
        if "mastery_counts" not in st.session_state:
            st.session_state["mastery_counts"] = { i: {"correct":0, "total":0} for i in range(len(chunks)) }

        st.header("🧪 Practice MCQs")
        idx = st.number_input("Select chunk to practice (1‑based)", 1, len(chunks), 1) - 1
        if mcqs[idx]:
            st.write(mcqs[idx])
            user_ans = st.text_input("Enter your answer option (e.g. A, B, C, D):")
            if st.button("Submit Answer for chunk"):
                correct = st.radio("Did you answer correctly?", ("Yes","No"))
                st.session_state["mastery_counts"][idx]["total"] += 1
                if correct == "Yes":
                    st.session_state["mastery_counts"][idx]["correct"] += 1
                correct_rate = (st.session_state["mastery_counts"][idx]["correct"] /
                                st.session_state["mastery_counts"][idx]["total"])
                st.success(f"Chunk {idx+1} mastery ≈ {correct_rate:.2f}")

        st.header("🔍 Topic Review Recommendation")
        mastery_rates = { i: (st.session_state["mastery_counts"][i]["correct"] /
                              st.session_state["mastery_counts"][i]["total"]
                              if st.session_state["mastery_counts"][i]["total"]>0 else 0.0)
                          for i in range(len(chunks)) }
        weak_order = sorted(mastery_rates, key=lambda i: mastery_rates[i])[:3]
        for w in weak_order:
            st.write(f"Chunk {w+1}: mastery approx {mastery_rates[w]:.2f}")

        st.header("❓ Ask a Question")
        question = st.text_area("Your question:", height=150)
        if st.button("Get Answer"):
            if "embeddings" in st.session_state:
                q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
                sims = (st.session_state["embeddings"] @ q_emb) / (
                        np.linalg.norm(st.session_state["embeddings"], axis=1) * np.linalg.norm(q_emb)
                )
                top3 = sims.argsort()[-3:][::-1]
                context = "\n\n".join([st.session_state["chunks"][i] for i in top3])

                sentiment_lbl = sentiment_pipe(question[:512])[0]["label"]
                style = "supportive, step‑by‑step" if sentiment_lbl=="NEGATIVE" else "clear and detailed"

                prompt = f"You are a professional tutor. Style: {style}. Context:\n{context}\n\nQuestion:\n{question}"
                answer = generate_text(prompt)

                prompt_flow = f"Explain the answer as a Mermaid flowchart. Context:\n{context}\n\nQuestion:\n{question}"
                flowchart = generate_text(prompt_flow)

                st.subheader("🧠 Answer")
                st.write(answer)
                st.subheader("📊 Flowchart Explanation")
                st.markdown("```mermaid\n" + flowchart + "\n```")
            else:
                st.error("Knowledge base not available — upload material first.")
