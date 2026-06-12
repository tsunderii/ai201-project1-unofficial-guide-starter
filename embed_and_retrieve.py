"""
Unofficial Guide to Ordering Matcha — Milestone 4: Embedding + Vector Store + Retrieval

This script handles the MIDDLE of the RAG pipeline:

    (already done) Ingestion -> Chunking
    (THIS FILE)    Embedding + Vector Store -> Retrieval
    (later)        Generation

It loads the chunks produced in Milestone 3, embeds them with sentence-transformers,
stores them in a ChromaDB collection, and provides a retrieve() function that we test
with the evaluation questions from planning.md.

It does NOT call an LLM or generate answers — that is the next milestone.

Run it with:
    python embed_and_retrieve.py
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Settings — kept in one place so they are easy to find and change.
# These match planning.md: all-MiniLM-L6-v2, ChromaDB, top-k = 4.
# ---------------------------------------------------------------------------
CHUNKS_PATH = "data/processed/chunks.json"   # input from Milestone 3
CHROMA_DIR = "chroma_db"                      # folder where ChromaDB persists data
COLLECTION_NAME = "matcha_guide"              # name of our vector collection
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"     # sentence-transformers model
TOP_K = 4                                     # how many chunks to retrieve per query

# Distance thresholds used by the inspection guide / warnings below.
# We use COSINE distance (0 = identical, 2 = opposite). Smaller is better.
GOOD_DISTANCE = 0.5    # "below this is a strong match"
WARN_DISTANCE = 0.6    # "above this is getting weak"
HIGH_DISTANCE = 0.7    # "above this is probably off-topic"

# The evaluation questions come straight from planning.md.
EVAL_QUESTIONS = [
    "What type of matcha drink would be best for a beginner who does not want something too bitter?",
    "What customizations can make a matcha drink taste sweeter or creamier?",
    "What is the difference between ceremonial and culinary matcha?",
]


# ===========================================================================
# STEP 1 — LOAD CHUNKS FROM THE PREVIOUS MILESTONE
# ===========================================================================
def load_chunks(path):
    """
    Load the list of chunk dictionaries written by the ingestion script.
    Each chunk should look like:
        {"text": "...", "source": "file.txt", "chunk_index": 0, "char_length": 469}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run the ingestion/chunking script first "
            f"so it creates data/processed/chunks.json."
        )

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


# ===========================================================================
# STEP 2 — VALIDATE THE CHUNKS BEFORE WE EMBED THEM
# ===========================================================================
def validate_chunks(chunks):
    """
    Check the chunks for common problems and print warnings. We do this BEFORE
    embedding so we catch bad input early instead of getting confusing results.

    Returns a clean list of chunks that are safe to embed.
    """
    print("=" * 70)
    print("VALIDATING CHUNKS")
    print("=" * 70)

    # Warning: no chunks loaded at all.
    if not chunks:
        print("  ⚠️  No chunks loaded! Check that the ingestion step ran and produced data.")
        return []

    good = []
    seen_ids = set()
    missing_metadata = 0
    empty_text = 0
    duplicate_ids = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source")
        chunk_index = chunk.get("chunk_index")

        # Warning: result/chunk that appears empty.
        if not text or not text.strip():
            empty_text += 1
            continue

        # Warning: missing metadata (we need source + chunk_index for attribution).
        if source is None or chunk_index is None:
            missing_metadata += 1
            continue

        # Build the unique id we'll use in ChromaDB and check for duplicates.
        chunk_id = f"{source}__{chunk_index}"
        if chunk_id in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(chunk_id)

        good.append(chunk)

    # Report what we found.
    if empty_text:
        print(f"  ⚠️  {empty_text} chunk(s) had empty text and were skipped.")
    if missing_metadata:
        print(f"  ⚠️  {missing_metadata} chunk(s) were missing source/chunk_index and were skipped.")
    if duplicate_ids:
        print(f"  ⚠️  {duplicate_ids} chunk(s) had duplicate ids (source + chunk_index) and were skipped.")

    print(f"  ✅ {len(good)} valid chunk(s) ready to embed "
          f"(out of {len(chunks)} loaded).")
    return good


# ===========================================================================
# STEP 3 — LOAD THE EMBEDDING MODEL
# ===========================================================================
def load_model():
    """Load the sentence-transformers model named in planning.md."""
    print("\nLoading embedding model:", EMBEDDING_MODEL_NAME)
    print("(The first run downloads the model; later runs use the cache.)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


# ===========================================================================
# STEP 4 + 5 + 6 — EMBED CHUNKS AND STORE THEM IN CHROMADB
# ===========================================================================
def build_collection(chunks, model):
    """
    Turn chunks into embeddings and store everything in a fresh ChromaDB
    collection. Returns the collection so retrieve() can query it.
    """
    # --- Create the embeddings ---------------------------------------------
    # model.encode() takes a list of texts and returns one vector per text.
    # We turn the numpy result into plain Python lists because that is what
    # ChromaDB expects.
    texts = [c["text"] for c in chunks]
    print(f"\nCreating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # --- Connect to ChromaDB -----------------------------------------------
    # PersistentClient saves the database to disk in CHROMA_DIR so the next
    # milestone can reuse it without re-embedding everything.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # --- Reset the collection ----------------------------------------------
    # Delete any old version so re-running this script gives a clean store
    # instead of stacking duplicate entries on top of old ones.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    # Create the collection. We tell ChromaDB to use COSINE distance, which
    # works well with all-MiniLM-L6-v2 and gives scores roughly in 0..2 where
    # smaller means more similar. (The default is L2, which would not match
    # the 0.5 / 0.6 / 0.7 thresholds we check below.)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Build the parallel lists ChromaDB wants ---------------------------
    ids = [f"{c['source']}__{c['chunk_index']}" for c in chunks]
    documents = texts
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "char_length": c.get("char_length", len(c["text"])),
        }
        for c in chunks
    ]

    # --- Add everything to the collection in one call ----------------------
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}'.")
    return collection, model


# ===========================================================================
# STEP 7 — THE RETRIEVAL FUNCTION
# ===========================================================================
def retrieve(query, collection, model, top_k=TOP_K):
    """
    Embed the query, search ChromaDB, and return + print the top_k chunks.

    Returns a list of result dicts, each with: source, chunk_index, distance, text.
    """
    # 1. Embed the query with the SAME model used for the chunks.
    query_embedding = model.encode([query]).tolist()

    # 2. Ask ChromaDB for the closest chunks. n_results = how many to return.
    #    We ask for documents/metadatas/distances so we can show attribution.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns "list of lists" (one inner list per query). We sent one
    # query, so we read index [0] from each parallel list.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Print and collect the results.
    print("\n" + "-" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)

    output = []
    if not documents:
        print("  ⚠️  No results returned — is the collection empty?")
        return output

    for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        result = {
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
            "text": doc,
        }
        output.append(result)

        print(f"\n  [{rank}] source={result['source']} "
              f"| chunk_index={result['chunk_index']} "
              f"| distance={dist:.4f}")
        # Show a short preview so the console stays readable; full text is in `result`.
        preview = doc.replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + " ..."
        print(f"      {preview}")

    return output


# ===========================================================================
# STEP 9 + 10 — INSPECTION GUIDE AND WARNINGS AFTER EACH QUERY
# ===========================================================================
def inspect_results(results):
    """
    Print a short, human-readable guide to judging the retrieval quality,
    plus automatic warnings for high distance scores or empty results.
    """
    print("\n  --- Inspection guide ---")

    # Warning: empty results.
    if not results:
        print("  ⚠️  No results to inspect.")
        return

    distances = [r["distance"] for r in results]
    best = min(distances)
    worst = max(distances)

    # Questions for YOU to answer by reading the chunks above:
    print("  • Are the returned chunks actually relevant to the question? (read them above)")
    print(f"  • Is the TOP distance below {GOOD_DISTANCE}? "
          f"best={best:.4f} -> {'YES ✅' if best < GOOD_DISTANCE else 'NO ⚠️'}")
    print("  • Do any chunks look off-topic or noisy? (e.g. wrong drink, leftover boilerplate)")

    # Automatic warnings on distance scores.
    if best > HIGH_DISTANCE:
        print(f"  ⚠️  Even the BEST distance ({best:.4f}) is above {HIGH_DISTANCE} — "
              f"likely off-topic. Your corpus may not cover this question yet.")
    elif best > WARN_DISTANCE:
        print(f"  ⚠️  Best distance ({best:.4f}) is above {WARN_DISTANCE} — weak match.")

    if worst > HIGH_DISTANCE:
        print(f"  ⚠️  At least one returned chunk has a high distance ({worst:.4f} > {HIGH_DISTANCE}) "
              f"— it may just be filler to reach top_k.")

    # Warning: a returned chunk is empty.
    if any(not r["text"].strip() for r in results):
        print("  ⚠️  One or more returned chunks are empty — check the stored data.")


# ===========================================================================
# MAIN — wire the steps together.
# ===========================================================================
def main():
    # Steps 1 + 2: load and validate chunks.
    chunks = load_chunks(CHUNKS_PATH)
    chunks = validate_chunks(chunks)
    if not chunks:
        print("\nNo valid chunks to embed. Stopping.")
        return

    # Step 3: load the model.
    model = load_model()

    # Steps 4-6: embed and store in ChromaDB.
    collection, model = build_collection(chunks, model)

    # Steps 7-9: test retrieval on the evaluation questions.
    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL ON EVALUATION QUESTIONS")
    print("=" * 70)

    for question in EVAL_QUESTIONS:
        results = retrieve(question, collection, model, top_k=TOP_K)
        inspect_results(results)


if __name__ == "__main__":
    main()
