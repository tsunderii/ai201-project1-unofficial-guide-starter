# The Unofficial Guide to Ordering Matcha — Project 1

## Project Overview

The Unofficial Guide to Ordering Matcha is a Retrieval-Augmented Generation (RAG)
system that answers beginner questions about ordering matcha drinks at specialty
tea and boba cafes. Rather than relying on a language model's general knowledge, the
system grounds every answer in a small, curated corpus of cafe menus, delivery
listings, community discussions, and matcha-education articles. A user question is
embedded, matched against a vector store of document chunks, and answered by an LLM
that is instructed to use only the retrieved text — or to decline when the corpus
does not cover the question. The goal is a guide that is both genuinely helpful to
newcomers and transparent about where its answers come from.

---

## Domain

Beginner’s guide to ordering matcha at specialty tea cafes — this system covers practical knowledge for people who are new to matcha and want to understand how to order matcha drinks from modern tea shops and boba-style cafes. This includes common drink types, sweetness levels, milk choices, cheese foam or cream toppings, iced vs. hot matcha, matcha quality, flavor profiles, and popular customizations.

This knowledge is valuable because specialty tea cafe menus often include matcha drinks with unfamiliar terms such as matcha latte, matcha cloud, jasmine matcha, salted cheese, ceremonial matcha, or culinary matcha. Official menus usually list drink names, ingredients, and prices, but they do not always explain what the drink tastes like, whether it is beginner-friendly, how strong the matcha flavor is, or how customers customize it. An unofficial guide can combine cafe menus, tea education articles, reviews, and community discussions to help beginners make better ordering decisions.

---

## Document Sources

The **final corpus uses 5 real-content source files.** I originally planned 10 sources, but several were blocked, JavaScript-rendered, or otherwise unavailable for clean text extraction (details below). Rather than pad the corpus with empty or fabricated files, I removed those and evaluated the system honestly on the 5 sources I could capture as real text.

| # | Source | Type | Local file (`data/raw/`) | URL |
|---|--------|------|--------------------------|-----|
| 1 | Molly Tea Official Menu | Official cafe menu | `01_mollytea_menu.txt` | https://www.mollytea.co.th/en/menu/ |
| 2 | Molly Tea USA Product Categories | Official cafe menu / product descriptions | `02_mollytea_usa_products.txt` | https://usa.mollytea.com/product-categories/ |
| 3 | HEYTEA Broadway Delivery Menu | Delivery menu / item descriptions | `06_heytea_broadway.txt` | https://postmates.com/store/heytea-broadway/EH_JrQSnUHWb6Xbjj-knWA |
| 4 | Naoki Matcha: Ceremonial vs Culinary Grade Matcha | Article / matcha grade guide | `08_naoki_matcha_grades.txt` | https://naokimatcha.com/blogs/articles/ceremonial-grade-matcha |
| 5 | JING Tea: Matcha Latte vs Traditional Matcha | Article / taste and texture guide | `10_jing_matcha_latte_vs_traditional.txt` | https://jingtea.com/blogs/matcha/matcha-latte-vs-traditional-matcha |

Each source above is saved as a plain `.txt` file in `data/raw/`, containing the meaningful text from the page (drink names, descriptions, matcha explanations, taste notes, and beginner advice), with navigation, ads, and cookie banners removed. Live scraping is intentionally avoided; the local-file approach keeps ingestion reproducible and easy to inspect. (Filenames keep their original numbering — `06`, `08`, `10` — so they trace back to the original plan.)

**Sources dropped from the original plan (and why):**

| Planned source | Reason it was dropped |
|----------------|-----------------------|
| Molly Tea Bellevue Delivery Menu (Grubhub) | JavaScript-rendered; no readable menu text could be extracted |
| HEYTEA Menu 2026 (`theheyteamenu.us`) | Returned HTTP 403 Forbidden |
| HEYTEA Supreme Matcha Latte (`theheyteamenu.us`) | Returned HTTP 403 Forbidden |
| Food & Wine: Ceremonial vs Culinary | Could not be fetched for clean text extraction |
| Epicurious: Beginner’s Guide to Matcha | Could not be fetched for clean text extraction |

The dropped HEYTEA pages are partly covered anyway: the HEYTEA Broadway delivery menu (source 3) includes the Cloud Matcha Latte and a supreme matcha latte, and the ceremonial-vs-culinary topic is covered by the Naoki Matcha article (source 4).

---

## Architecture Summary

The system is a five-stage RAG pipeline:

1. **Document Ingestion** (`ingest_and_chunk.py`) loads every `.txt` file from `data/raw/` into a consistent in-memory structure.
2. **Cleaning & Chunking** strips HTML tags, unescapes entities (e.g. `&amp;`, `&#39;`, `&nbsp;`), removes navigation/footer/cookie boilerplate, then splits each document into ~500-character chunks with 100-character overlap, preferring paragraph and sentence boundaries so menu items and review comments stay intact. Chunks are written with metadata to `data/processed/chunks.json`.
3. **Embedding & Vector Store** (`embed_and_retrieve.py`) encodes each chunk with the `all-MiniLM-L6-v2` sentence-transformer and stores the vectors, text, and metadata in a persistent ChromaDB collection configured for cosine distance.
4. **Retrieval** embeds the user query with the same model and returns the top `k = 4` nearest chunks, each carrying its source filename, chunk index, and distance score for attribution.
5. **Grounded Generation** (`query.py`) formats the retrieved chunks as labeled context and sends them, with a strict system prompt, to the Groq `llama-3.3-70b-versatile` model. A Gradio interface (`app.py`) exposes the whole pipeline as a question box that returns an answer plus the source files used.

```
Document Ingestion → Cleaning & Chunking → Embedding + Vector Store → Retrieval → Grounded Generation → Gradio UI
```

---

## Chunking Strategy

**Chunk size:** ~500 characters.

**Overlap:** 100 characters.

**Why these choices fit the documents:** the corpus mixes very short text (individual menu items and product descriptions) with longer educational article paragraphs. A 500-character target keeps each chunk focused on one drink, one menu item, or one matcha concept, so retrieval returns tight, on-topic context rather than several unrelated drinks at once. The 100-character overlap protects information that spans a boundary — for example, a sentence naming a drink's ingredients followed by a sentence on whether it is beginner-friendly — so a single retrieved chunk still carries enough context to be useful.

**Preprocessing before chunking:** HTML tags are removed with a regular expression; HTML entities are unescaped; non-breaking spaces are normalized; lines matching known navigation/footer/cookie/advertisement patterns are dropped; and runs of whitespace are collapsed. The chunker then prefers to end a chunk at a paragraph break, then a sentence end, then a newline, then a space, falling back to a hard 500-character cut only when no natural boundary is nearby.

**Final chunk count:** 26 chunks across the 5 source files in `data/raw/` (Molly Tea menu: 3, Molly Tea USA products: 4, HEYTEA Broadway: 2, Naoki Matcha grades: 8, JING latte-vs-traditional: 9). This is a small corpus by design, given that several planned sources could not be captured as clean text; the chunking and retrieval behavior is identical to what a larger corpus would use.

---

## Sample Chunks

Five representative chunks produced by the pipeline, one drawn from each source document currently ingested. Each is labeled with its source filename, its index within that document, and its character length.

**Chunk 1 — source: `01_mollytea_menu.txt` (chunk 0, 376 chars)**
```
Molly Tea — Matcha Beverages (Official Menu)
Premium Jasmine Matcha
Ingredients: Green Tea + Matcha Cheese + Fresh Milk. Served cold or hot.
Fragrant jasmine milk tea served over ice, topped with a smooth layer of rich matcha cheese. Combines floral, creamy, and earthy notes.
White Champaca Matcha
Ingredients: Green Tea + Matcha Cheese + Fresh Milk. Served cold or hot.
```

**Chunk 2 — source: `02_mollytea_usa_products.txt` (chunk 0, 335 chars)**
```
Molly Tea USA — Matcha Products
White Champaca Matcha
A premium matcha beverage featuring carefully selected matcha powder blended with Anchor cream into a dense, silky matcha cheese. Paired with rare Bailan-scented tea, this drink combines deep matcha flavor with subtle orchid notes and a bamboo leaf aroma.
Premium Jasmine Matcha
```

**Chunk 3 — source: `06_heytea_broadway.txt` (chunk 0, 480 chars)**
```
HEYTEA (Broadway, New York) — Matcha Menu
Cloud Matcha Latte — $9.99
1000 Mesh Matcha + real dairy milk + Original Cheese Cloud. A matcha latte topped with a layer of cheese cloud (a creamy cheese foam). Contains coconut and milk products.
Triple Supreme Matcha Latte — $9.99
Matcha Cloud + Matcha + Matcha Mochi + Matcha Jelly + real milk. A matcha-forward drink layering a matcha cheese cloud, matcha latte base, chewy matcha mochi, and matcha jelly. Contains milk products.
```

**Chunk 4 — source: `08_naoki_matcha_grades.txt` (chunk 0, 385 chars)**
```
Naoki Matcha — Ceremonial vs Culinary Grade Matcha
Understanding the grade classification
Outside Japan, matcha consumers typically divide matcha into ceremonial and culinary grades. However, Japan itself does not use this classification system. There is no standardized, government-regulated grading system for matcha, which makes these labels inherently subjective and unregulated.
```

**Chunk 5 — source: `10_jing_matcha_latte_vs_traditional.txt` (chunk 0, 355 chars)**
```
JING Tea — Matcha Latte vs Traditional Matcha
Core difference
Traditional matcha consists of powder whisked with hot water only, while a matcha latte combines matcha powder with steamed or frothed milk and often includes sweetness. This single addition of milk fundamentally transforms the drink's flavor, texture, and cultural context.
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, producing 384-dimensional embeddings, with cosine distance configured in ChromaDB. It was chosen because it is small, fast, runs locally with no API cost, and performs well on short, informal English text — which fits a corpus of menu snippets and Reddit comments.

**Production tradeoff reflection:** If this were deployed for real users and cost were not a constraint, I would weigh a larger or API-hosted embedding model with higher accuracy and a longer context window, and — because the corpus includes informal language and non-English tea terms — better support for multilingual and domain-specific text. The tradeoff is latency and cost: a stronger model improves retrieval relevance but is slower and more expensive per query, which matters for an interactive interface.

---

## Retrieval Test Results

Three test queries run directly against the retrieval layer (before generation), each showing the top 4 chunks returned with their source, chunk index, and cosine distance (lower = more similar).

**Query 1: "What makes a matcha drink less bitter for a beginner?"**
| Rank | Source | Chunk | Distance |
|------|--------|-------|----------|
| 1 | `10_jing_matcha_latte_vs_traditional.txt` | 1 | 0.350 |
| 2 | `08_naoki_matcha_grades.txt` | 7 | 0.416 |
| 3 | `10_jing_matcha_latte_vs_traditional.txt` | 8 | 0.417 |
| 4 | `10_jing_matcha_latte_vs_traditional.txt` | 2 | 0.419 |

*Why these are relevant:* The top chunk is the JING passage explaining that milk proteins and fats soften matcha's bitterness and create creaminess — directly answering how to make a drink less bitter. The Naoki chunk (rank 2) adds that ceremonial-grade matcha has lower bitterness and needs less sweetener, and the JING chunk at rank 4 explains how milk dilutes the tea's intensity. All four chunks are about milk, dilution, and bitterness, which is exactly the topic of the query.

**Query 2: "What is the difference between ceremonial and culinary matcha?"**
| Rank | Source | Chunk | Distance |
|------|--------|-------|----------|
| 1 | `08_naoki_matcha_grades.txt` | 0 | 0.325 |
| 2 | `08_naoki_matcha_grades.txt` | 3 | 0.352 |
| 3 | `10_jing_matcha_latte_vs_traditional.txt` | 0 | 0.481 |
| 4 | `08_naoki_matcha_grades.txt` | 2 | 0.500 |

*Why these are relevant:* Three of the four chunks come from the Naoki Matcha grades article, the single most on-topic source in the corpus. Rank 1 introduces the ceremonial-vs-culinary classification, rank 2 covers the practical "drinking vs recipes" distinction, and rank 4 explains why the labels are unregulated and unreliable — together these cover both the factual difference and the important caveat. The JING chunk at rank 3 is a weaker match (0.48) because it discusses grades only in passing.

**Query 3: "How is a matcha latte different from traditional matcha?"**
| Rank | Source | Chunk | Distance |
|------|--------|-------|----------|
| 1 | `10_jing_matcha_latte_vs_traditional.txt` | 8 | 0.194 |
| 2 | `10_jing_matcha_latte_vs_traditional.txt` | 0 | 0.232 |
| 3 | `10_jing_matcha_latte_vs_traditional.txt` | 5 | 0.292 |
| 4 | `08_naoki_matcha_grades.txt` | 7 | 0.313 |

These are the strongest distances of the three tests (0.19–0.31), and all four chunks are on-topic — the top three are from the JING article that compares latte and traditional matcha preparation, flavor, and texture directly.

---

## Grounded Generation

**System prompt grounding instruction:** the model is given a system prompt that states it may answer *only* from the provided context chunks, must not use outside or general matcha knowledge, and must reply with the exact sentence "I don't have enough information on that." when the context is insufficient. The generation call uses `temperature=0` to keep answers deterministic and discourage creative additions.

**How source attribution is surfaced in the response:** sources are appended programmatically from the retrieval metadata, not invented by the LLM. After generation, the system collects the `source` filename of each retrieved chunk (de-duplicated, best-match first) and returns it alongside the answer; the Gradio UI shows these under "Retrieved from." When the model returns the refusal, no sources are attached, because the answer is not grounded in any chunk.

---

## Example Responses

Two grounded answers with source attribution visible in the output, followed by one out-of-scope query showing the refusal.

**Example 1 (in scope, with sources)**
> **Question:** What customizations can make a matcha drink taste sweeter or creamier?
>
> **Answer:** To make a matcha drink taste sweeter or creamier, you can add honey or syrups for sweetness. For creaminess, you can use milk, as milk proteins and fats soften the bitterness and create creaminess. Different milk types, such as oat milk, almond milk, or dairy, can also contribute distinct flavors and creaminess to your matcha drink. Additionally, even unsweetened milk introduces natural sweetness.
>
> **Sources:** `10_jing_matcha_latte_vs_traditional.txt`, `08_naoki_matcha_grades.txt`, `01_mollytea_menu.txt`

**Example 2 (in scope, with sources)**
> **Question:** What's the difference between a Cloud Matcha and a regular matcha latte?
>
> **Answer:** A Cloud Matcha Latte, as seen in the HEYTEA menu, is topped with a layer of cheese cloud, which is a creamy cheese foam, whereas a regular matcha latte, as described, typically consists of matcha powder combined with steamed or frothed milk.
>
> **Sources:** `10_jing_matcha_latte_vs_traditional.txt`, `06_heytea_broadway.txt`

**Example 3 (out of scope, refusal)**
> **Question:** What parking situation is best near UCSC?
>
> **Answer:** I don't have enough information on that.
>
> **Sources:** *(none — the answer is not grounded in any retrieved chunk)*

The corpus contains nothing about parking, so retrieval returned only weakly-related chunks (best distance 0.86) and the grounding rule produced the refusal instead of a guess.

*Note: the LLM is not perfectly deterministic even at `temperature=0`, so the exact wording of an answer may vary slightly between runs. The retrieved sources and the substance of each answer stay stable; only phrasing shifts.*

---

## Query Interface

The interface is a Gradio web app (`app.py`) launched with `python app.py` and served at `http://localhost:7860`.

**Input field**
- *Your question* — a single-line textbox where the user types a matcha-ordering question. The query can be submitted by clicking the **Ask** button or by pressing **Enter** in the textbox.

**Output fields**
- *Answer* — a multi-line textbox showing the grounded answer generated from the retrieved chunks (or the refusal sentence when the corpus does not cover the question).
- *Retrieved from* — a multi-line textbox listing the source filenames the answer was drawn from, as a bulleted list. It shows a "no sources" note when the system refuses.

**Sample interaction transcript**
```
Your question:  What customizations can make a matcha drink taste sweeter or creamier?

[Ask]

Answer:
To make a matcha drink taste sweeter or creamier, you can add honey or syrups for
sweetness. For creaminess, you can use milk, as milk proteins and fats soften the
bitterness and create creaminess. Different milk types, such as oat milk, almond milk,
or dairy, can also contribute distinct flavors and creaminess to your matcha drink.
Additionally, even unsweetened milk introduces natural sweetness.

Retrieved from:
• 10_jing_matcha_latte_vs_traditional.txt
• 08_naoki_matcha_grades.txt
• 01_mollytea_menu.txt
```

---

## Evaluation Report

> **Scope note:** The results below were produced by running the live system against the final corpus of 5 source files in `data/raw/` (26 chunks). Distances are cosine distance (lower = more similar).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Best matcha drink for a beginner avoiding bitterness | Matcha latte / sweeter milk-based; milk and sweetness smooth it | A matcha latte made with ceremonial-grade matcha, which has lower bitterness and needs less sweetener | Relevant (0.42) | **Accurate** |
| 2 | Customizations for a sweeter/creamier drink | Sweetness, creamy milk (oat), latte-style, cream/foam toppings | Add honey or syrups for sweetness; use oat, almond, or dairy milk for creaminess; unsweetened milk also adds natural sweetness | Relevant (0.28) | **Accurate** |
| 3 | Difference between ceremonial and culinary matcha | Quality/use difference, **and labels are inconsistent, unregulated marketing terms not to over-trust** | Ceremonial is best for drinking plain; culinary works best in recipes | Relevant (0.33) | **Partially accurate** |
| 4 | Difference between a Cloud Matcha and a regular matcha latte | Cloud = latte + cheese foam topping; latte = matcha with milk | Cloud Matcha is topped with a creamy cheese-cloud foam; a regular latte is frothed milk over a matcha shot without the cheese cloud | Relevant (0.34) | **Accurate** |
| 5 | What to order for a stronger matcha flavor | Matcha-forward drink, less sugar, fewer toppings, avoid milk overpowering | Order traditional matcha (whisked with water only), since milk dilution masks the flavor | Relevant (0.36) | **Accurate** |

**Robustness check (off-topic input):** asked "What parking situation is best near UCSC?", the system returned "I don't have enough information on that." with no sources — and the best retrieved chunk sat at distance 0.86, far outside the ~0.3–0.4 range of the answerable questions. This confirms the grounding layer refuses out-of-domain questions instead of hallucinating.

**Summary:** four accurate answers and one partially accurate, with no hallucinations across the real corpus. The single partial result (Q3) is a genuine generation-stage limitation analyzed below, not a retrieval failure — the missing information was present in the retrieved context.

---

## Failure Case Analysis

**Question that failed:** Q3 — "What is the difference between ceremonial and culinary matcha?"

**What the system returned:** A correct but minimal description of the practical difference ("ceremonial grade matcha is best suited for drinking plain, while culinary grade matcha works best in recipes"), but it omitted the most beginner-relevant point in the expected answer: that "ceremonial" and "culinary" are unregulated, inconsistent marketing labels that beginners should not over-trust.

**Root cause (tied to a specific pipeline stage):** This is a **generation-stage** failure, not a retrieval failure. The Naoki Matcha source (`08_naoki_matcha_grades.txt`) explicitly states that "there is no standardized, government-regulated grading system for matcha" and that the labels are "inherently subjective and unregulated" — and that source was the top-ranked retrieval for this question (cosine distance 0.33). Retrieval did its job: the caveat was inside the context window. With four chunks of context and an instruction to be "clear and beginner-friendly," the model compressed its answer toward the most concrete, actionable distinction (plain vs. recipes) and discarded the more abstract caveat during synthesis.

**Why that stage caused the failure:** Language models compress multi-chunk context toward the most salient, concrete points when asked to be concise. The caveat was the most conceptually important part of the answer but the least concrete, so it was the first thing dropped. Because retrieval succeeded, no amount of retrieval tuning would fix this — the fix has to happen at generation.

**What I would change to fix it:** Tighten the system prompt to require preserving caveats and qualifications — for example, "If the context warns about a common misconception or says a term is unreliable, include that warning in your answer." This is a single, testable change isolated to the generation stage. A secondary option is to reduce `top_k` so the model has fewer competing points to compress, though that risks dropping useful context elsewhere.

---

## Spec Reflection

**One way the spec helped me during implementation:** Writing the Chunking Strategy and Retrieval Approach sections of `planning.md` *before* coding gave me concrete parameters — 500-character chunks, 100-character overlap, `all-MiniLM-L6-v2`, and top-k = 4 — and, more importantly, the *reasoning* behind them. Because I had already articulated that my corpus mixes short reviews with longer articles, I could justify a chunker that prefers paragraph and sentence boundaries instead of a naive fixed split, and I never had to stop mid-implementation to re-decide basic parameters. The spec acted as a contract I could implement against directly.

**One way my implementation diverged from the spec, and why:** My requirements file originally pinned `sentence-transformers==3.4.1`, but adding the Gradio interface in the final milestone forced a change: Gradio 6.x requires `huggingface-hub >= 1.2`, which in turn requires `transformers >= 5.0`, which requires `sentence-transformers >= 5.0`. I therefore upgraded the embedding stack to 5.x. I verified this was safe by re-running retrieval before and after the upgrade and confirming the cosine distances were identical, since the underlying `all-MiniLM-L6-v2` model weights are the same across versions. A second, smaller divergence: `planning.md` did not specify a distance metric, and ChromaDB defaults to squared-L2; I configured cosine distance instead so my relevance thresholds were interpretable.

---

## AI Usage

**Instance 1 — implementing the chunker**
- *What I gave the AI:* my Chunking Strategy section from `planning.md` (500-character chunks, 100-character overlap, "preserve paragraphs, menu items, and review comments when possible") and my list of document types.
- *What it produced:* a `chunk_text()` function using a sliding character window that snaps each chunk's end to the nearest paragraph break, then sentence end, then newline, then space, plus an inspection step printing representative and random chunks and warnings for empty, short, or HTML-containing chunks.
- *What I changed, verified, or overrode:* I inspected the representative and random chunks myself and checked the short-chunk warnings before accepting the 500/100 parameters, rather than assuming they were correct. *(Edit this line to match what you actually adjusted — e.g. whether you tuned the boundary look-back distance, the short-chunk threshold, or the sample text.)*

**Instance 2 — keeping the evaluation honest**
- *What I gave the AI:* I asked Claude to act as a reviewer for my evaluation and README, with instructions to point out weaknesses rather than just generate text.
- *What it produced:* it flagged that my system was still running on starter sample files rather than the full set of sources listed in my README, warned that reporting the evaluation as-is would misrepresent the project and make "Accurate" results circular, and identified that one of my evaluation questions was a methodology question my content corpus could not answer.
- *What I changed, verified, or overrode:* I decided to revise that evaluation question (Q4) into a content question my documents could answer, and I combined the critique with my own judgment about which fix preserved the intent of my evaluation plan rather than accepting a rewrite wholesale.

**Instance 3 — refusing to pad the corpus with empty files**
- *What I gave the AI:* a revised list of 10 sources to use, several of which turned out to be blocked, JavaScript-rendered, or unavailable for clean text extraction. The AI had created empty placeholder files in `data/raw/` for the sources it could not capture.
- *What it produced:* a working pipeline plus those empty stub files, which it was prepared to leave in place as "pending" sources while the README continued to describe a 10-source corpus.
- *What I changed, verified, or overrode:* I overrode Claude's suggestion to keep empty placeholder source files because I realized they would make the corpus misleading. Instead, I removed the empty stubs and kept only the 5 files with real source content, then updated the README to describe the smaller corpus honestly.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your GROQ_API_KEY to a .env file (see .env.example)

# 3. Ingest and chunk the documents in data/raw/
python ingest_and_chunk.py

# 4. Build the embeddings and vector store, and test retrieval
python embed_and_retrieve.py

# 5. Test grounded generation from the command line
python query.py

# 6. Launch the Gradio interface
python app.py   # then open http://localhost:7860
```
