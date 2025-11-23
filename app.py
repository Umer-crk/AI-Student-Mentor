import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# pyKT imports
from pykt.models import DKT  # Deep Knowledge Tracing model
from pykt.preprocessing import ResponseSequence, KCMapping

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="AI Study Mentor (Stable KT)", page_icon="📘", layout="wide")

groq_api_key = st.sidebar.text_input("Groq API Key (optional, for better generation)", type="password")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    # Generation model (local)
    gen_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name)
    gen_pipe = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer)

    # Sentiment
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Embeddings
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    return gen_pipe, sentiment, embed

gen_pipe, sentiment_pipe, embed_model = load_models()

# ------------------------------
# Initialize Knowledge Tracing
# ------------------------------
@st.cache_resource
def init_kt_model(num_kcs):
    # Use DKT with given number of knowledge components
    model = DKT(n_items=num_kcs, hidden_size=32, n_layers=1)  # small for performance
    # We need a KC mapping: we'll just treat each chunk index as a KC
    return model

# ------------------------------
# Utility Functions
# ------------------------------
def call_groq(prompt, max_tokens=700):
    if not groq_api_key:
        return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"Groq API failed: {e}")
        return None

def generate_text(prompt):
    api_out = call_groq(prompt)
    if api_out:
        return api_out
    out = gen_pipe(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"]

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
st.title("📘 AI Study Mentor — Professional with Knowledge Tracing")

uploaded = st.file_uploader("Upload PDF / Image (for study content)", type=["pdf","jpg","png","jpeg"])
max_pages = st.sidebar.slider("Max pages to OCR", 5, 200, 50)

if uploaded:
    with st.spinner("Extracting text..."):
        full_text = ocr_extract(uploaded, max_pages=max_pages)
    if not full_text.strip():
        st.error("No text extracted.")
    else:
        chunks = chunk_text(full_text)
        st.success(f"Text split into {len(chunks)} chunks for study.")

        # Generate study material for each chunk
        notes = [generate_text(f"Summarize into study notes:\n\n{ch}") for ch in chunks]
        mcqs = [generate_text(f"Generate 5 MCQs with 4 options + answer from:\n\n{ch}") for ch in chunks]
        flows = [generate_text(f"Convert into a Mermaid flowchart:\n\n{ch}") for ch in chunks]

        st.header("📝 Notes")
        for i, n in enumerate(notes):
            st.subheader(f"Chunk {i+1}")
            st.write(n)

        st.header("❓ MCQs")
        for i, m in enumerate(mcqs):
            st.subheader(f"Chunk {i+1}")
            st.write(m)

        st.header("📊 Flowchart (by chunk)")
        for i, f in enumerate(flows):
            st.subheader(f"Chunk {i+1}")
            st.markdown("```mermaid\n" + f + "\n```")

        # Build embeddings for retrieval
        embs = embed_model.encode(chunks, convert_to_numpy=True)
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embs

        # Initialize KT model if not in session
        if "kt_model" not in st.session_state:
            st.session_state["kt_model"] = init_kt_model(len(chunks))
            # Also prepare a response sequence
            st.session_state["seq"] = ResponseSequence()

        # Practice MCQs to update KT
        st.header("🧪 Practice MCQs")
        chunk_index = st.number_input("Select chunk to practice (1‑based)", 1, len(chunks), 1) - 1
        if mcqs[chunk_index]:
            st.write(mcqs[chunk_index])
            user_ans = st.text_input("Type your answer (e.g. A, B, C, D)")
            if st.button("Submit your MCQ answer"):
                # Ask user if correct (very simple)
                correct = st.radio("Is your answer correct?", ("Yes", "No"))
                obs = 1 if correct == "Yes" else 0
                # Add to sequence
                kc_map = KCMapping({i: i for i in range(len(chunks))})
                st.session_state["seq"].add_response(kc_map, chunk_index, obs)
                # Fit / update DKT
                st.session_state["kt_model"].fit(st.session_state["seq"])
                mastery = st.session_state["kt_model"].predict(st.session_state["seq"])
                # mastery shape: (#sequences, num_kcs)
                master_probs = mastery[0]  # only one student
                st.session_state["mastery"] = master_probs.tolist()
                st.success(f"Updated mastery: {master_probs}")

        # Recommend weak topics
        if "mastery" in st.session_state:
            st.header("🔍 Topics to Review (Lowest Mastery)")
            probs = st.session_state["mastery"]
            weakest = sorted(range(len(probs)), key=lambda i: probs[i])[:3]
            for w in weakest:
                st.write(f"Chunk {w+1}: mastery ≈ {probs[w]:.2f}")

        # Q&A with retrieval + sentiment + flowchart
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

                prompt = f"You are a helpful tutor. Style: {style}. Context: {context}\n\nQuestion: {question}"
                answer = generate_text(prompt)

                prompt_flow = f"Explain this answer with a mermaid flowchart. Context: {context}\n\nQuestion: {question}"
                flow = generate_text(prompt_flow)

                st.subheader("🧠 Answer")
                st.write(answer)
                st.subheader("📊 Explanation Flowchart")
                st.markdown("```mermaid\n" + flow + "\n```")
            else:
                st.error("Knowledge base not initialized (upload first).")
