"""
Unofficial Guide to Ordering Matcha — Milestone 3: Ingestion + Cleaning + Chunking

This script handles the FIRST part of the RAG pipeline only:

    Document Ingestion -> Cleaning -> Chunking -> Inspection/Debug output

It deliberately does NOT do embeddings, the vector store, retrieval, or generation.
Those come in later milestones. Everything here is plain standard-library Python so
it is easy to read, run, and debug.

Run it with:
    python ingest_and_chunk.py

It reads every .txt file in data/raw/ and writes the cleaned, chunked result to
data/processed/chunks.json so later milestones can load it for embeddings.
"""

import html
import json
import os
import random
import re

# ---------------------------------------------------------------------------
# Settings — change these in one place if you adjust your chunking strategy.
# These match planning.md: 500-character chunks with 100-character overlap.
# ---------------------------------------------------------------------------
RAW_DIR = "data/raw"                       # folder containing the input .txt files
PROCESSED_DIR = "data/processed"           # folder where chunks.json is written
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "chunks.json")

CHUNK_SIZE = 500                           # target characters per chunk
CHUNK_OVERLAP = 100                        # characters shared between neighboring chunks
SHORT_CHUNK_THRESHOLD = 50                 # chunks shorter than this look "suspiciously short"

# A fixed seed so the "5 random chunks" are the same every run.
# This makes debugging repeatable. Remove the seed if you want true randomness.
RANDOM_SEED = 42


# ===========================================================================
# STEP 1 — DOCUMENT INGESTION
# ===========================================================================
def load_documents(raw_dir):
    """
    Load every .txt file from `raw_dir` into a consistent structure.

    Returns a list of dictionaries, one per document, each shaped like:
        {"source": "sample_menu_mollytea.txt", "raw_text": "....."}

    We keep the RAW text untouched here so that, while debugging, we can always
    compare the original text against the cleaned text and see exactly what the
    cleaning step changed.
    """
    documents = []

    # Make sure the folder exists so we can give a clear error instead of crashing.
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Could not find the raw documents folder: '{raw_dir}'. "
            f"Create it and add some .txt files first."
        )

    # sorted() keeps the order stable so output is the same run to run.
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.lower().endswith(".txt"):
            continue  # skip anything that is not a .txt file

        path = os.path.join(raw_dir, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        documents.append({"source": filename, "raw_text": raw_text})

    return documents


# ===========================================================================
# STEP 2 — CLEANING
# ===========================================================================

# Lines that are obviously navigation / footer / cookie / ad boilerplate.
# If a line is *mostly* one of these phrases, we drop the whole line.
# Keep this list lowercase; matching is done in lowercase.
BOILERPLATE_KEYWORDS = [
    "accept all cookies",
    "we use cookies",
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "subscribe to our newsletter",
    "advertisement",
    "order online",
    "vote | share | report",
    "home | menu | locations",
]


def strip_html_tags(text):
    """Remove anything that looks like an HTML/XML tag, e.g. <p>, </div>, <br/>."""
    return re.sub(r"<[^>]+>", " ", text)


def unescape_entities(text):
    """
    Turn HTML entities into normal characters.
    html.unescape handles named entities (&amp; &nbsp; &mdash; &quot;) AND
    numeric ones (&#39; -> '). &nbsp; becomes a non-breaking space (\\xa0),
    which we convert to a normal space in the whitespace step below.
    """
    return html.unescape(text)


def looks_like_boilerplate(line):
    """Return True if a single line is mostly navigation/footer/cookie/ad text."""
    lowered = line.lower()
    return any(keyword in lowered for keyword in BOILERPLATE_KEYWORDS)


def clean_text(raw_text):
    """
    Clean one document. Order matters here:

      1. Strip HTML tags        (<p>, </div>, etc.)
      2. Unescape HTML entities (&amp; -> &, &#39; -> ', &nbsp; -> space)
      3. Drop boilerplate lines (cookie banners, footers, ads, nav menus)
      4. Normalize whitespace   (collapse runs of spaces, trim blank lines)

    We keep the useful content: drink names, menu descriptions, customer
    opinions, ratings, and matcha explanations all survive because we only
    remove markup and obvious junk lines, never normal sentences.
    """
    # 1. Remove HTML tags first so tag contents don't get glued to words.
    text = strip_html_tags(raw_text)

    # 2. Convert entities like &amp; &nbsp; &#39; into real characters.
    text = unescape_entities(text)

    # Replace non-breaking spaces (from &nbsp;) with normal spaces.
    text = text.replace("\xa0", " ")

    # 3. Go line by line, dropping boilerplate and empty lines.
    cleaned_lines = []
    for line in text.split("\n"):
        # Collapse runs of spaces/tabs within the line into a single space.
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            continue  # skip empty lines
        if looks_like_boilerplate(line):
            continue  # skip cookie/footer/nav/ad lines

        cleaned_lines.append(line)

    # 4. Rejoin the kept lines with a BLANK line between them. The blank line
    #    ("\n\n") is a paragraph break, which the chunker prefers as a place to
    #    end a chunk — that is what keeps menu items and review comments whole.
    text = "\n\n".join(cleaned_lines)

    return text.strip()


# ===========================================================================
# STEP 3 — CHUNKING
# ===========================================================================
def _find_break_point(text, window_start, hard_end):
    """
    Helper for chunk_text(). Given a slice text[window_start:hard_end], find a
    "nice" place to end the chunk so we don't cut a sentence or paragraph in
    half. We look (in priority order) for the LAST:
        1. paragraph break "\\n\\n"
        2. sentence end ". " / "! " / "? "
        3. single newline
        4. space
    ...but only if that break is not too early (we don't want a tiny chunk).

    Returns the absolute index in `text` where the chunk should end.
    If no good break is found, we fall back to a hard cut at `hard_end`.
    """
    slice_ = text[window_start:hard_end]

    # Don't accept a break in the first ~60% of the window, otherwise chunks
    # get too short. We only "snap back" within the last 40% of the window.
    min_keep = int((hard_end - window_start) * 0.6)

    # Try each kind of boundary from strongest (paragraph) to weakest (space).
    for boundary in ["\n\n", ". ", "! ", "? ", "\n", " "]:
        idx = slice_.rfind(boundary)
        if idx != -1 and idx >= min_keep:
            # +len(boundary) so the boundary characters stay with this chunk.
            return window_start + idx + len(boundary)

    # No good boundary found — just cut at the hard limit.
    return hard_end


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split `text` into overlapping chunks of about `chunk_size` characters.

    Strategy (matches planning.md):
      * Target size is 500 characters, with 100 characters of overlap so a
        thought that spans a boundary still appears whole in one chunk.
      * We PREFER to end a chunk at a paragraph break, then a sentence end,
        then a newline, then a space — so menu items, reviews, and paragraphs
        stay readable and mostly self-contained instead of being cut mid-word.

    Returns a list of chunk strings (no metadata yet — that is added later).
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        hard_end = min(start + chunk_size, n)

        if hard_end < n:
            # Not the last chunk: try to snap to a natural boundary.
            end = _find_break_point(text, start, hard_end)
        else:
            # Last chunk: just take whatever is left.
            end = hard_end

        chunk = text[start:end].strip()
        if chunk:  # never store empty chunks
            chunks.append(chunk)

        if end >= n:
            break  # we've consumed the whole document

        # Move the window forward, stepping back by `overlap` characters.
        next_start = end - overlap

        # Safety: always make forward progress, otherwise we could loop forever
        # (e.g. if a boundary landed right at the overlap distance).
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_document(document):
    """
    Turn ONE loaded+cleaned document into a list of chunk records with metadata.

    Each chunk record looks like:
        {
            "source": "sample_menu_mollytea.txt",  # which file it came from
            "chunk_index": 0,                       # position within that file
            "char_length": 412,                     # how many characters
            "text": "Matcha Latte 'classic' ..."    # the actual chunk text
        }
    """
    cleaned = clean_text(document["raw_text"])
    pieces = chunk_text(cleaned)

    records = []
    for i, piece in enumerate(pieces):
        records.append(
            {
                "source": document["source"],
                "chunk_index": i,
                "char_length": len(piece),
                "text": piece,
            }
        )
    return records


# ===========================================================================
# STEP 4 — INSPECTION / DEBUG OUTPUT
# ===========================================================================
def print_separator(title):
    """Small helper to make the console output easy to scan."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_one_cleaned_document(documents):
    """Print the cleaned version of the first document so you can eyeball it."""
    if not documents:
        return
    first = documents[0]
    print_separator(f"CLEANED DOCUMENT (for inspection): {first['source']}")
    print(clean_text(first["raw_text"]))


def show_representative_chunks(chunks, how_many=5):
    """
    Print `how_many` chunks spread evenly across the whole list, so you see a
    mix of early, middle, and late chunks rather than just the first few.
    """
    print_separator(f"{how_many} REPRESENTATIVE CHUNKS (evenly spaced)")
    if not chunks:
        print("(no chunks)")
        return

    count = min(how_many, len(chunks))
    # Pick evenly spaced positions: 0, n/5, 2n/5, ...
    step = max(1, len(chunks) // count)
    positions = [min(i * step, len(chunks) - 1) for i in range(count)]

    for pos in positions:
        c = chunks[pos]
        print(f"\n--- chunk #{pos} | source={c['source']} "
              f"| index_in_file={c['chunk_index']} | length={c['char_length']} ---")
        print(c["text"])


def show_random_chunks(chunks, how_many=5):
    """Print `how_many` randomly chosen chunks (reproducible via RANDOM_SEED)."""
    print_separator(f"{how_many} RANDOM CHUNKS (seed={RANDOM_SEED})")
    if not chunks:
        print("(no chunks)")
        return

    rng = random.Random(RANDOM_SEED)
    count = min(how_many, len(chunks))
    sample_positions = rng.sample(range(len(chunks)), count)

    for pos in sample_positions:
        c = chunks[pos]
        print(f"\n--- chunk #{pos} | source={c['source']} "
              f"| index_in_file={c['chunk_index']} | length={c['char_length']} ---")
        print(c["text"])


# ===========================================================================
# STEP 5 — SANITY CHECKS / WARNINGS
# ===========================================================================
def run_checks(documents, chunks):
    """
    Print warnings for common ingestion/chunking problems. None of these stop
    the script — they are signals telling you whether to adjust your cleaning
    or chunking strategy.
    """
    print_separator("SANITY CHECKS / WARNINGS")

    warnings = []

    # 1. Empty chunks (should be impossible since we skip them, but verify).
    empty = [c for c in chunks if not c["text"].strip()]
    if empty:
        warnings.append(f"{len(empty)} EMPTY chunk(s) found — cleaning may be too aggressive.")

    # 2. Chunks that still contain HTML tags — cleaning missed something.
    html_like = [c for c in chunks if re.search(r"<[^>]+>", c["text"])]
    if html_like:
        warnings.append(
            f"{len(html_like)} chunk(s) still contain HTML-like tags "
            f"(e.g. in {html_like[0]['source']}) — improve strip_html_tags()."
        )

    # 3. Suspiciously short chunks — may be fragments that won't retrieve well.
    short = [c for c in chunks if c["char_length"] < SHORT_CHUNK_THRESHOLD]
    if short:
        warnings.append(
            f"{len(short)} chunk(s) are shorter than {SHORT_CHUNK_THRESHOLD} chars "
            f"— could be junk fragments or over-splitting."
        )

    # 4. Too few chunks overall — maybe not enough source documents.
    if len(chunks) < 50:
        warnings.append(
            f"Only {len(chunks)} chunks total (< 50). Add more source documents "
            f"to data/raw/ for a useful guide."
        )

    # 5. Too many chunks overall — maybe chunk size is too small.
    if len(chunks) > 2000:
        warnings.append(
            f"{len(chunks)} chunks total (> 2000). Consider larger chunks or "
            f"fewer/smaller documents to keep embedding fast and cheap."
        )

    if warnings:
        for w in warnings:
            print(f"  ⚠️  {w}")
    else:
        print("  ✅ No issues detected.")


# ===========================================================================
# STEP 6 — SAVE OUTPUT
# ===========================================================================
def save_chunks(chunks, output_path):
    """Write all chunk records to a JSON file for the next milestone to load."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print_separator("SAVED")
    print(f"  Wrote {len(chunks)} chunks to {output_path}")


# ===========================================================================
# MAIN — tie all the steps together in pipeline order.
# ===========================================================================
def main():
    # 1. Ingest
    documents = load_documents(RAW_DIR)

    # 2 + 3. Clean and chunk every document, collecting all chunks together.
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document))

    # 4. Inspection output
    show_one_cleaned_document(documents)
    show_representative_chunks(all_chunks, how_many=5)
    show_random_chunks(all_chunks, how_many=5)

    print_separator("TOTALS")
    print(f"  Total documents loaded: {len(documents)}")
    print(f"  Total chunks created:   {len(all_chunks)}")

    # 5. Warnings
    run_checks(documents, all_chunks)

    # 6. Save for later milestones (embeddings/vector store)
    save_chunks(all_chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()
