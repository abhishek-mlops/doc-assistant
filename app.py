"""Document Assistant - RAG Chatbot

Streamlit + Groq (openai/gpt-oss-120b) + sentence-transformers + FAISS.
No LangChain, no LlamaIndex, no external vector DB.
"""

import hashlib
import io
import os
import re

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
LLM_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 8
EMBED_BATCH_SIZE = 32

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 6rem; max-width: 920px; }
    [data-testid="stSidebar"] { min-width: 280px; }
    .hint-row .stButton > button {
        width: 100%;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.35);
        background: transparent;
        font-weight: 500;
    }
    .hint-row .stButton > button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    header [data-testid="stToolbar"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_groq_api_key():
    """Read GROQ_API_KEY from st.secrets, falling back to os.environ."""
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        # No secrets file or key missing -> fall back to environment variable.
        pass
    return os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedder():
    """Load the sentence-transformers model once per server process."""
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


def extract_text_from_pdf(file_bytes, progress_cb=None):
    """Extract text from every page of a PDF. Returns the full text."""
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
        if progress_cb:
            progress_cb((i + 1) / max(total_pages, 1))
    return "\n\n".join(pages_text)


def sliding_window(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a long string into `size`-char windows with `overlap`-char overlap."""
    pieces = []
    step = size - overlap
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
        start += step
    return pieces


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split by paragraphs first, then into ~500-char chunks with 50-char overlap.

    - Paragraphs are separated by blank lines.
    - Short paragraphs are merged together until a chunk would exceed `size`.
    - Paragraphs longer than `size` are cut into overlapping windows.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        cleaned = re.sub(r"\s+", " ", p).strip()
        if cleaned:
            paragraphs.append(cleaned)

    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(para) > size:
            # Flush whatever is buffered, then window the long paragraph.
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(sliding_window(para, size, overlap))
        elif not buffer:
            buffer = para
        elif len(buffer) + 1 + len(para) <= size:
            buffer = buffer + " " + para
        else:
            chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)

    return chunks


def embed_texts(embedder, texts, progress_cb=None):
    """Embed texts in batches; returns a float32 array of L2-normalised vectors."""
    all_vecs = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        vecs = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_vecs.append(vecs)
        if progress_cb:
            progress_cb(min(start + EMBED_BATCH_SIZE, total) / total)
    return np.vstack(all_vecs).astype("float32")


def build_faiss_index(embeddings):
    """Inner-product index on normalised vectors == cosine similarity."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def is_broad_question(question):
    q = question.lower()
    return any(
        phrase in q
        for phrase in (
            "summar",
            "overview",
            "key point",
            "main point",
            "important term",
            "this document",
            "this pdf",
            "what is this",
        )
    )


def retrieve(question, k=TOP_K):
    """Return a list of (chunk_number, chunk_text, score) for the top-k chunks."""
    chunks = st.session_state.chunks
    embeddings = np.asarray(st.session_state.embeddings, dtype="float32")
    if embeddings.ndim != 2 or len(chunks) == 0:
        return []

    # Rebuild FAISS from stored vectors. The C++ index itself does not survive Streamlit reruns.
    index = build_faiss_index(embeddings)
    embedder = load_embedder()
    q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    if is_broad_question(question):
        k = min(max(k, 12), len(chunks))
    k = min(k, len(chunks))
    scores, ids = index.search(q_vec, k)

    results = []
    seen = set()
    for score, idx in zip(scores[0], ids[0]):
        idx = int(idx)
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        results.append((idx + 1, chunks[idx], float(score)))

    # Summaries need the start of the PDF, not only the closest 8 snippets.
    if is_broad_question(question):
        for idx, text in enumerate(chunks[:8]):
            if idx not in seen:
                seen.add(idx)
                results.append((idx + 1, text, 0.0))
    return results


def build_prompt(question, retrieved):
    context = "\n\n".join(f"[Chunk {num}] {text}" for num, text, _ in retrieved)
    return (
        "The passages below are excerpts from the user's uploaded PDF.\n"
        "Answer the question using these passages. Combine them if needed. Cite chunk numbers.\n"
        "If they only partly match, still answer from what is there.\n"
        "Say you could not find it only when the passages are clearly about something else.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}"
    )


def ask_groq(client, prompt):
    kwargs = dict(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help users understand an uploaded document. "
                    "Use the given passages. Be specific. Do not refuse a summary or overview "
                    "when passages from the document are provided."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    try:
        kwargs["reasoning_effort"] = "low"
        response = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return (content or "").strip()


def reset_document_state():
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.file_hash = None
    st.session_state.file_name = None
    st.session_state.messages = []


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

api_key = get_groq_api_key()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Document")
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if st.session_state.file_name:
        st.caption(st.session_state.file_name)
        st.metric("Chunks", len(st.session_state.chunks))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Remove PDF", use_container_width=True, disabled=not st.session_state.chunks):
            reset_document_state()
            st.rerun()

    if not api_key:
        st.error("Add GROQ_API_KEY in secrets to chat.")

# ----------------------------------------------------------------------------
# Index upload
# ----------------------------------------------------------------------------
if not api_key:
    st.stop()

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.md5(file_bytes).hexdigest()

    if current_hash != st.session_state.file_hash:
        reset_document_state()
        progress = st.progress(0.0, text="Reading PDF...")
        try:
            raw_text = extract_text_from_pdf(
                file_bytes,
                progress_cb=lambda f: progress.progress(f * 0.3, text="Reading PDF..."),
            )
        except Exception as e:
            progress.empty()
            st.error(f"Could not read this PDF: {e}")
            st.stop()

        if not raw_text or not raw_text.strip():
            progress.empty()
            st.warning("This PDF has no selectable text. Try a text PDF, not a scan.")
            st.stop()

        progress.progress(0.3, text="Indexing...")
        chunks = split_text(raw_text)
        if not chunks:
            progress.empty()
            st.warning("No usable text found in this PDF.")
            st.stop()

        embedder = load_embedder()
        embeddings = embed_texts(
            embedder,
            chunks,
            progress_cb=lambda f: progress.progress(0.35 + f * 0.6, text="Indexing..."),
        )
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.file_hash = current_hash
        st.session_state.file_name = uploaded_file.name
        progress.empty()
        st.rerun()

# ----------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------
st.title("Document Assistant")

ready = st.session_state.embeddings is not None and len(st.session_state.chunks) > 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources · chunks {', '.join(str(n) for n, _, _ in msg['sources'])}"):
                for num, text, score in msg["sources"]:
                    st.markdown(f"**Chunk {num}** · {score:.2f}")
                    st.text(text)

if ready and not st.session_state.messages:
    st.markdown('<div class="hint-row">', unsafe_allow_html=True)
    hint_cols = st.columns(3)
    for col, hint in zip(
        hint_cols,
        ("Summarize this document", "What are the key points?", "List important terms"),
    ):
        if col.button(hint, use_container_width=True):
            st.session_state.pending_question = hint
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

question = st.chat_input(
    "Ask about the PDF…" if ready else "Upload a PDF in the sidebar to start",
    disabled=not ready,
)
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    if not ready:
        st.warning("Upload a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                retrieved = retrieve(question, TOP_K)
                if not retrieved:
                    answer = "The document was indexed, but no matching text was retrieved. Try another question."
                else:
                    prompt = build_prompt(question, retrieved)
                    client = get_groq_client(api_key)
                    answer = ask_groq(client, prompt)
                    if not answer:
                        answer = "The model returned an empty reply. Please try the question again."
            except Exception as e:
                retrieved = []
                answer = f"Could not generate an answer: `{e}`"

        st.markdown(answer)
        if retrieved:
            with st.expander(f"Sources · chunks {', '.join(str(n) for n, _, _ in retrieved)}"):
                for num, text, score in retrieved:
                    st.markdown(f"**Chunk {num}** · {score:.2f}")
                    st.text(text)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved})
