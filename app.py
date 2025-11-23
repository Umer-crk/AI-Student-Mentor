import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests
import io
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Knowledge tracing
from pyBKT.models import Model as BKTModel

# ================================================
# CONFIG
# ================================================
st.set_page_config(page_title="AI Study Mentor", page_icon="📘", layout="wide")

# Sidebar: API key for LLM (Groq-style API)
groq_api_key = st.sidebar.text_input("Your Groq API Key (optional)", type="password")

# ===== Load Free Models =====
@st.cache_resource(show_spinner=False)
def load_models():
    # Generation / summarization model
    gen_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name)
    gen_pipe = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer)

    # Sentiment analysis
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Embedding model
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    return gen_pipe, sentiment, embed

gen_pipe, sentiment_pipe, embed_model = load_models()

# ===== Knowledge Tracing Model =====
@st.cache_resource(show_spinner=False)
def init_bkt_model():
    # We'll treat each “chunk” as a “knowledge component” (KC)
    # Simple BKT with default parameters
    bkt = BKTModel(seed=42)
    # Initialize with 1 KC; we'll adapt number of KC to chunks dynamically
    bkt.add_kcs([0])  # dummy initialization
    return bkt

bkt_model = init_bkt_model()

# ================================================
# UTILITIES
# ================================================
def call_groq(prompt, max_tokens=700):
    if not groq_api_key:
        return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            timeout=60
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"Groq API error: {e}")
        return None

def generate_text(prompt):
    # Try API first
    api_out = call_groq(prompt)
    if api_out:
        return api_out
    # Fallback to local
    res = gen_pipe(prompt, max_length=512, do_sample=False)
    return res[0]["generated_text"]

def extract_text(uploaded_file, dpi=150, max_pages=50):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    text = ""
    if name.endswith(".pdf"):
        data = uploaded_file.read()
        pages = convert_from_bytes(data, dpi=dpi)
        pages = pages[:max_pages]
        for p in pages:
            text += "\n" + pytesseract.image_to_string(p)
    else:
        img = Image.open(uploaded_file)
        text = pytesseract.image_to_string(img)
    return text

def chunk_text(text, max_chars=2000):
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len > max_chars:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ================================================
# UI
# ================================================
st.title("📘 AI Study Mentor — Professional (Free‑friendly)")
st.markdown("""
This mentor uses **Knowledge Tracing**, **RAG**, and **Emotion‑aware tutoring**  
to help you learn more effectively.
""")

uploaded = st.file_uploader("Upload PDF or Image for Study Material", type=["pdf","png","jpg","jpeg"])
max_pages = st.sidebar.slider("Max pages to OCR", 5, 200, 50)

if uploaded:
    with st.spinner("Extracting text via OCR..."):
        text = extract_text(uploaded, max_pages=max_pages)
    if not text.strip():
        st.error("Could not extract any text.")
    else:
        chunks = chunk_text(text)
        st.success(f"Text split into {len(chunks)} conceptual chunks.")

        # Generate Notes, MCQs, Flowchart for each chunk
        notes = []
        mcqs = []
        flows = []
        for i, ch in enumerate(chunks):
            notes.append(generate_text(f"Summarize into structured study notes:\n\n{ch}"))
            mcqs.append(generate_text(f"Generate 5 MCQs with options + answers from:\n\n{ch}"))
            flows.append(generate_text(f"Convert this into a mermaid flowchart with nodes & decisions:\n\n{ch}"))

        st.header("📝 Notes")
        for i, n in enumerate(notes):
            st.subheader(f"Chunk {i+1}")
            st.write(n)

        st.header("❓ MCQs")
        for i, m in enumerate(mcqs):
            st.subheader(f"Chunk {i+1}")
            st.write(m)

        st.header("📊 Flowcharts (per chunk)")
        for i, f in enumerate(flows):
            st.subheader(f"Chunk {i+1}")
            st.markdown("```mermaid\n" + f + "\n```")

        # Build embeddings of chunks
        embs = embed_model.encode(chunks, convert_to_numpy=True)
        st.session_state["chunk_texts"] = chunks
        st.session_state["chunk_embeddings"] = embs

        # Initialize student mastery if not done
        if "mastery" not in st.session_state:
            # Start mastery for each chunk as 0.2 (initial guess)
            st.session_state["mastery"] = [0.2] * len(chunks)
            # Re-init BKT with correct number of KCs
            bkt_model = init_bkt_model()
            bkt_model.add_kcs(list(range(len(chunks))))
            st.session_state["bkt_model"] = bkt_model

        st.success("Study material ready. Now you can answer MCQs or ask questions.")

        # MCQ practice
        st.header("🧪 Practice MCQs")
        q_index = st.number_input("Select chunk number to practice", min_value=1, max_value=len(chunks), step=1)
        idx = q_index - 1
        st.write("**MCQs for this chunk:**")
        st.write(mcqs[idx])

        user_answer = st.text_input("Enter your selected option (e.g. A, B, C, D):")
        if st.button("Submit Answer"):
            # Here we *don’t know correct answer* because it's generated; we could parse MCQs,
            # but for simplicity, ask the user whether they got it right:
            correct = st.radio("Did you answer correctly?", ("Yes", "No"))
            # Update knowledge tracing
            kc_id = idx
            obs = 1 if correct == "Yes" else 0
            bkt = st.session_state["bkt_model"]
            bkt.update(data={ "0": [(kc_id, obs)] })  # student id "0"
            mastery = bkt.predict(data={ "0": [(kc_id, )] })  # next-step prediction
            new_mastery = mastery["skill_predictions"][0][kc_id]
            st.session_state["mastery"][idx] = new_mastery
            st.success(f"Updated mastery for chunk {idx+1}: {new_mastery:.2f}")

        # Recommend next topic
        st.header("🔍 Recommended Topics to Review")
        mastery = st.session_state["mastery"]
        worst = sorted(range(len(mastery)), key=lambda x: mastery[x])[:3]
        st.write([f"Chunk {i+1}: mastery {mastery[i]:.2f}" for i in worst])

        # Q&A with context (RAG)
        st.header("❓ Ask a Question About Your Material")
        question = st.text_area("Enter your question here:", height=150)
        if st.button("Ask Mentor"):
            if "chunk_texts" in st.session_state:
                # retrieve by embedding similarity
                q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
                sims = (st.session_state["chunk_embeddings"] @ q_emb) / (
                    np.linalg.norm(st.session_state["chunk_embeddings"], axis=1)
                    * np.linalg.norm(q_emb)
                )
                # pick top 3 relevant chunks
                top3 = sims.argsort()[-3:][::-1]
                context = "\n\n".join([st.session_state["chunk_texts"][i] for i in top3])

                # sentiment
                sent = sentiment_pipe(question[:512])[0]["label"]
                style = "step-by-step and supportive" if sent == "NEGATIVE" else "clear and detailed"

                prompt = f"You are a professional tutor. Use style: {style}. Context: {context}\n\nQuestion: {question}"
                answer = generate_text(prompt)

                # Also generate explanation flowchart
                prompt_flow = f"Explain the answer as a mermaid flowchart. Context: {context}\n\nQuestion: {question}"
                flowchart = generate_text(prompt_flow)

                st.subheader("🧠 Answer")
                st.write(answer)
                st.subheader("📊 Explanation Flowchart")
                st.markdown("```mermaid\n" + flowchart + "\n```")
            else:
                st.error("No study material loaded yet.")

