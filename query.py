"""
Unofficial Guide to Ordering Matcha — Milestone 5 (part 1): Grounded Generation

This file connects the retrieval step you already built to a Groq LLM and
produces a GROUNDED answer — meaning the model may only use the chunks we
retrieved, never outside knowledge.

Pipeline position:
    Ingestion -> Chunking -> Embedding + Vector Store -> Retrieval -> (THIS FILE) Generation

It reuses retrieve() from embed_and_retrieve.py exactly as-is. It does NOT
re-embed or re-chunk anything; it just queries the ChromaDB collection that
embed_and_retrieve.py already built and persisted in chroma_db/.

Public function: ask(question) -> dict with keys: answer, sources, retrieved_chunks
"""

import os

from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# Reuse the retrieval function and settings from the previous milestone.
# (Importing this module is safe: its heavy work is under `if __name__ == ...`.)
from embed_and_retrieve import (
    retrieve,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
load_dotenv()                              # reads .env so GROQ_API_KEY is available
GROQ_MODEL = "llama-3.3-70b-versatile"     # the LLM named in planning.md

# The EXACT sentence the model must use when the context is not enough.
# We check for this text afterwards to decide whether to attach sources.
REFUSAL_MESSAGE = "I don't have enough information on that."

# The grounding prompt. This is the single most important piece for preventing
# hallucination: it tells the model to use ONLY the provided context and to
# refuse otherwise. We send it as the "system" role so it frames every answer.
SYSTEM_PROMPT = f"""You are a careful assistant for a beginner's guide to ordering matcha at specialty tea cafes.

Follow these rules strictly:
1. Answer ONLY using the information in the provided context chunks.
2. Do NOT use outside knowledge, general matcha knowledge, or guesses.
3. If the context does not contain enough information to answer the question,
   reply with EXACTLY this sentence and nothing else:
   "{REFUSAL_MESSAGE}"
4. Keep the answer clear and beginner-friendly.
"""


# ---------------------------------------------------------------------------
# Lazy, cached resources — loaded once, then reused on every call.
# This keeps each ask() fast and gives clear errors if something is missing.
# ---------------------------------------------------------------------------
_model = None
_collection = None
_groq_client = None


def _get_resources():
    """Load the embedding model, the ChromaDB collection, and the Groq client once."""
    global _model, _collection, _groq_client

    # 1. Embedding model — must be the SAME one used to build the collection.
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2. ChromaDB collection — connect to the store embed_and_retrieve.py created.
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' was not found in "
                f"'{CHROMA_DIR}'. Run `python embed_and_retrieve.py` first to "
                f"build the vector store."
            )

    # 3. Groq client — needs GROQ_API_KEY from .env.
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Add it to your .env file "
                "(get a free key at https://console.groq.com)."
            )
        _groq_client = Groq(api_key=api_key)

    return _model, _collection, _groq_client


def _format_context(chunks):
    """
    Turn the retrieved chunk dicts into a single context string for the LLM.
    Each block is labeled with its source so the model can ground its answer.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        label = f"[Chunk {i}] (source: {c['source']}, chunk_index: {c['chunk_index']})"
        blocks.append(f"{label}\n{c['text']}")
    return "\n\n".join(blocks)


def ask(question, top_k=TOP_K):
    """
    Answer a question using ONLY the retrieved chunks (grounded generation).

    Returns a dictionary:
        {
            "answer": str,             # the LLM's grounded answer (or the refusal)
            "sources": [str, ...],     # source filenames, taken from retrieval metadata
            "retrieved_chunks": [...]  # the raw chunk dicts that were retrieved
        }
    """
    model, collection, groq_client = _get_resources()

    # 1. RETRIEVE — reuse your existing retrieval function unchanged.
    retrieved_chunks = retrieve(question, collection, model, top_k=top_k)

    # If retrieval found nothing at all, we cannot ground an answer.
    if not retrieved_chunks:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "retrieved_chunks": []}

    # 2. FORMAT the retrieved chunks into a context block.
    context = _format_context(retrieved_chunks)

    # 3. BUILD the message the user role sends: context + the actual question.
    user_prompt = (
        f"Context chunks:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )

    # 4. CALL the Groq LLM. temperature=0 makes the answer deterministic and
    #    discourages creative (hallucinated) additions.
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    # 5. ATTACH SOURCES PROGRAMMATICALLY from the retrieval metadata — not from
    #    whatever the LLM might claim. We dedupe while keeping retrieval order
    #    (best match first). If the model refused (not enough info), we attach
    #    no sources, because the answer is not grounded in any chunk.
    if REFUSAL_MESSAGE.lower().rstrip(".") in answer.lower():
        sources = []
    else:
        sources = []
        for c in retrieved_chunks:
            if c["source"] not in sources:
                sources.append(c["source"])

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }


# ---------------------------------------------------------------------------
# Quick end-to-end test. Run:  python query.py
# The 4th question should trigger the refusal because our matcha documents
# do not contain anything about parking near UCSC.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEST_QUESTIONS = [
        "What type of matcha drink would be best for a beginner who does not want something too bitter?",
        "What customizations can make a matcha drink taste sweeter or creamier?",
        "What is the difference between ceremonial and culinary matcha?",
        "What parking situation is best near UCSC?",  # expected: refusal
    ]

    for q in TEST_QUESTIONS:
        result = ask(q)
        print("\n" + "#" * 70)
        print("QUESTION:", q)
        print("#" * 70)
        print("ANSWER:", result["answer"])
        print("SOURCES:", result["sources"] if result["sources"] else "(none — not grounded)")
