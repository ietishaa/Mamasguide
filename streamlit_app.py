"""MamaGuide: guideline-grounded maternal health assistant (Streamlit app).

Same pipeline as MamaGuide_RAG_Safety_Evals.ipynb:
rule-based red-flag check -> FAISS retrieval over the WHO ANC guideline -> grounded Gemini answer (JSON).
"""
import json
import os
import re
import time

import faiss
import numpy as np
import requests
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_PATH = "who_anc_guideline.pdf"
PDF_URL = "https://ncbi.nlm.nih.gov/books/NBK409108/pdf/Bookshelf_NBK409108.pdf"
CHUNK_SIZE, OVERLAP, TOP_K = 900, 150, 4
REPO_URL = "https://github.com/ietishaa/Mamasguide"

st.set_page_config(page_title="MamaGuide", page_icon="🤰", layout="centered")

# ---------------- Rule layer ----------------
RED_FLAGS = {
    "heavy bleeding": r"\b(heavy|a lot of|soaking).{0,20}bleed|bleeding heavily",
    "severe headache / vision": r"(severe|worst|bad).{0,15}headache|blurr(ed|y) vision|seeing (spots|flashes)",
    "seizure / fainting": r"seizure|convuls|faint(ed|ing)|passed out",
    "severe abdominal pain": r"(severe|intense|sharp).{0,20}(abdominal|stomach|belly) pain",
    "reduced fetal movement": r"(baby|fetus).{0,25}(not moving|stopped moving|moving less)|reduced (fetal|baby) movement",
    "breathing / chest": r"can'?t breathe|short(ness)? of breath|chest pain",
    "fluid leaking": r"(water broke|leaking fluid|gush of fluid)",
    "high fever": r"(high )?fever|temperature of 3[89]",
    "self-harm thoughts": r"hurt(ing)? myself|end my life|don'?t want to (live|be here)|harm (my|the) baby",
}
ESCALATION_MSG = ("⚠️ This may need urgent medical attention. Please contact your doctor or midwife now, "
                  "or call emergency services (911 in the U.S.) if symptoms are severe. "
                  "If you're having thoughts of harming yourself, call or text 988.")


def rule_check(query):
    q = query.lower()
    return [name for name, pat in RED_FLAGS.items() if re.search(pat, q)]


# ---------------- Index (built once, cached) ----------------
@st.cache_resource(show_spinner="Loading the WHO guideline and building the search index (first load takes about a minute)...")
def build_index():
    if not os.path.exists(PDF_PATH):
        resp = requests.get(PDF_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        resp.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            f.write(resp.content)

    chunks = []
    for page_num, page in enumerate(PdfReader(PDF_PATH).pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE]
            if len(piece) > 50:
                chunks.append({"page": page_num, "text": piece})
            start += CHUNK_SIZE - OVERLAP

    embedder = SentenceTransformer(EMBED_MODEL)
    vecs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                           batch_size=64, show_progress_bar=False)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.asarray(vecs, dtype="float32"))
    return embedder, index, chunks


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def retrieve(question, k=TOP_K):
    embedder, index, chunks = build_index()
    q = embedder.encode([question], normalize_embeddings=True)
    _, ids = index.search(np.asarray(q, dtype="float32"), k)
    return [chunks[i] for i in ids[0]]


def llm(prompt):
    client = get_client()
    for attempt in range(2):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            if "429" in str(e) and attempt == 0:
                time.sleep(20)
                continue
            raise


# ---------------- Grounded generation ----------------
PROMPT = """You are a maternal health information assistant. Use ONLY the guideline excerpts below.

Rules:
- If the question describes symptoms that could be urgent or dangerous, set "escalate": true.
- If the excerpts do not contain the answer, set "in_scope": false and do not guess.
- Cite the page numbers you used.
- Plain language, under 120 words. Never diagnose or prescribe.

Return ONLY JSON:
{{"answer": "...", "pages": [int], "escalate": bool, "in_scope": bool, "reason": "..."}}

Guideline excerpts:
{context}

Question: {question}"""


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def ask(question, use_rules=True, k=TOP_K):
    flags = rule_check(question) if use_rules else []
    if flags:
        return {"answer": ESCALATION_MSG, "pages": [], "escalate": True, "in_scope": True,
                "layer": "rules", "reason": ", ".join(flags), "context": ""}
    hits = retrieve(question, k)
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    out = parse_json(llm(PROMPT.format(context=context, question=question)))
    out.setdefault("answer", ""); out.setdefault("pages", [])
    out.setdefault("escalate", False); out.setdefault("in_scope", True)
    out["layer"] = "llm"; out["context"] = context
    if out["escalate"]:
        out["answer"] = ESCALATION_MSG
    return out


# ---------------- UI ----------------
def render_answer(r):
    if r["escalate"]:
        st.error(r["answer"])
        source = "safety rule" if r["layer"] == "rules" else "model judgment"
        st.caption(f"Escalated by {source}" + (f": {r['reason']}" if r.get("reason") and r["layer"] == "rules" else ""))
    elif not r["in_scope"]:
        st.info("The WHO antenatal care guideline doesn't cover this question, so I can't answer it. "
                "Try asking about antenatal care, nutrition, supplements, or tests during pregnancy.")
    else:
        st.markdown(r["answer"])
        if r["pages"]:
            pages = ", ".join(str(p) for p in sorted(set(r["pages"])))
            st.caption(f"Source: WHO Recommendations on Antenatal Care (2016), pages {pages}")


with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "MamaGuide answers questions using only the **WHO Recommendations on Antenatal Care "
        "for a Positive Pregnancy Experience (2016)**.\n\n"
        "- Urgent symptoms are caught by safety rules before the model runs\n"
        "- Answers cite guideline pages\n"
        "- Out-of-scope questions are declined"
    )
    st.markdown(f"[View code and evaluation on GitHub]({REPO_URL})")
    st.divider()
    st.caption("Research demo, not medical advice. Always consult a healthcare provider.")

st.title("MamaGuide")
st.markdown("Ask a question about pregnancy care. Answers come from the WHO antenatal care guideline, with page citations.")

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.markdown("**Try asking:**")
    examples = [
        "How many antenatal care contacts are recommended?",
        "What is recommended about iron and folic acid?",
        "Is physical activity recommended during pregnancy?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["content"])

question = st.chat_input("Ask about antenatal care...")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the guideline..."):
                result = ask(question)
            render_answer(result)
            st.session_state.history.append({"role": "assistant", "content": result})
        except Exception as e:
            if "429" in str(e):
                st.warning("The free API quota is busy right now. Please wait a minute and try again.")
            else:
                st.warning("Something went wrong generating an answer. Please try again.")
