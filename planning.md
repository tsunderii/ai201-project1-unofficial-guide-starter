# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

chose the domain beginner’s guide to ordering matcha at specialty tea cafes. This domain focuses on helping people who are new to matcha understand how to order matcha drinks from modern tea shops, boba cafes, and specialty drink shops.

This knowledge is valuable because matcha cafe menus can be confusing for beginners. Menus often include terms like matcha latte, cloud matcha, salted cheese, jasmine matcha, ceremonial matcha, culinary matcha, sweetness level, and milk alternatives without explaining what they actually mean.

---

## Documents

> **Revision note (final stage):** The original source list leaned on Reddit and Grubhub pages that were blocked or hard to capture reliably. I revised it toward official cafe menus and matcha-education articles. During final capture, several of even the revised sources turned out to be blocked, JavaScript-rendered, or unavailable for clean text extraction, so the **final corpus uses the 5 sources below** that I could capture as real text. I removed the empty placeholder files rather than pad the corpus, and evaluated honestly on these 5.

**Final corpus (5 sources used):**

| # | Source | Type | Local file (`data/raw/`) | URL |
|---|--------|------|--------------------------|-----|
| 1 | Molly Tea Official Menu | Official cafe menu | `01_mollytea_menu.txt` | https://www.mollytea.co.th/en/menu/ |
| 2 | Molly Tea USA Product Categories | Official cafe menu / product descriptions | `02_mollytea_usa_products.txt` | https://usa.mollytea.com/product-categories/ |
| 3 | HEYTEA Broadway Delivery Menu | Delivery menu / item descriptions | `06_heytea_broadway.txt` | https://postmates.com/store/heytea-broadway/EH_JrQSnUHWb6Xbjj-knWA |
| 4 | Naoki Matcha: Ceremonial vs Culinary Grade Matcha | Article / matcha grade guide | `08_naoki_matcha_grades.txt` | https://naokimatcha.com/blogs/articles/ceremonial-grade-matcha |
| 5 | JING Tea: Matcha Latte vs Traditional Matcha | Article / taste and texture guide | `10_jing_matcha_latte_vs_traditional.txt` | https://jingtea.com/blogs/matcha/matcha-latte-vs-traditional-matcha |

**Planned sources that were dropped (could not be captured as clean text):** Molly Tea Bellevue (Grubhub, JavaScript-rendered); HEYTEA Menu 2026 and HEYTEA Supreme Matcha Latte (`theheyteamenu.us`, HTTP 403); Food & Wine ceremonial-vs-culinary article; Epicurious beginner’s guide.


---

## Chunking Strategy

**Chunk size:**
I will split my documents into chunks of about 500 characters with an overlap of about 100 characters. This chunk size fits my documents because my corpus includes a mix of short menu descriptions, short reviews, Reddit comments, and longer educational articles. A smaller chunk size helps keep each retrieved chunk focused on one drink, one opinion, or one matcha concept.

**Overlap:**
The 100-character overlap helps because some useful information may span across two chunks. For example, one sentence might describe a drink’s ingredients, while the next sentence explains its flavor or whether it is beginner-friendly. The overlap increases the chance that a retrieved chunk still contains enough context to be useful.

**Reasoning:**
If my chunks are too small, the system might retrieve fragments that do not fully answer the question. For example, it might retrieve only “salted cheese topping” without explaining whether it makes the drink sweeter, creamier, or better for beginners. If my chunks are too large, the system might retrieve too much unrelated information at once, such as multiple drinks or multiple opinions in one chunk, making it harder for the LLM to give a focused answer. Bad retrieval results would look like answers that are too vague, mix up drinks, or cite chunks that only partially relate to the question.

---

## Retrieval Approach

**Embedding model:**
I plan to use all-MiniLM-L6-v2 through sentence-transformers as my embedding model. I will retrieve the top 4 chunks per query. Four chunks should give the LLM enough context to compare multiple sources without overwhelming it with too much unrelated text.

**Top-k:**
Retrieving 4 chunks should give the LLM enough information to answer with context from multiple sources without including too much unrelated text. If I retrieve too few chunks, the system might miss useful information or rely on only one source. For example, a question about beginner-friendly matcha might need information from both a matcha education article and customer opinions. If I retrieve too many chunks, the context could become noisy and include unrelated drinks, general boba opinions, or off-topic comments.

**Production tradeoff reflection:**
If I were deploying this for real users and cost was not a constraint, I would consider a stronger embedding model with better accuracy, longer context length, and better support for informal language. Since some documents may include Reddit comments, menu terms, or non-English tea names, I would also consider multilingual support and domain-specific accuracy. The tradeoff is that larger embedding models can be slower and more expensive.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What type of matcha drink would be best for a beginner who does not want something too bitter? | A matcha latte or a sweeter milk-based matcha drink would be best because milk and sweetness can make the matcha flavor smoother and less bitter. |
| 2 | What customizations can make a matcha drink taste sweeter or creamier? | Adding sweetness, choosing a creamy milk like oat milk, ordering a latte-style drink, or adding cream/foam toppings can make the drink sweeter or creamier. |
| 3 | What is the difference between ceremonial and culinary matcha? | Ceremonial matcha is usually marketed as higher quality and better for drinking plain, while culinary matcha is usually used for lattes, baking, or mixing. However, the labels can be inconsistent, so beginners should not rely only on those terms. |
| 4 | What is the difference between a Cloud Matcha and a regular matcha latte? | A Cloud Matcha is a matcha latte topped with a layer of salted cheese foam, which adds a creamy, slightly savory layer that balances the matcha's bitterness. A regular matcha latte is stone-ground matcha shaken with fresh milk at a chosen sweetness level. The Cloud version is often considered more approachable for first-timers because the foam softens the flavor. |
| 5 | What should someone order if they want a stronger matcha flavor? | They should choose a drink where matcha is the main focus, use less sugar, avoid too many toppings or fruit flavors, and avoid drinks where milk or cream overpowers the matcha. |

> **Revision note (evaluation stage):** Question 4 originally asked "Why are Reddit discussions and customer reviews useful for this guide?" During evaluation I found this was a *methodology* question about the project itself, not a question my content corpus could answer — retrieval returned only weakly-related chunks (cosine distance ~0.9) and the system correctly refused. I replaced it with a content question that exercises my stated risk of confusing similar drinks, so the question is answerable from the documents and the result is meaningful evidence.

---

## Anticipated Challenges

1. One risk I anticipated was noisy or inconsistent documents, since I originally planned to include customer opinions, Reddit comments, and reviews where different people describe the same drink differently (one person calls a drink too bitter, another calls it smooth). In practice the final corpus ended up being cafe menus and matcha-education articles rather than community reviews, so this particular risk was reduced — though the menus and articles can still disagree (for example on how much milk softens bitterness).

2. Another risk is off-topic retrieval or poor chunking. Some cafe menus and discussion threads may include non-matcha drinks, general boba drinks, or unrelated cafe opinions. If the chunks are too large or split badly, the system might retrieve information about the wrong drink or miss important context, such as the drink name, ingredients, or whether the drink is beginner-friendly.

---

## Architecture

The "Unofficial Guide to Ordering Matcha" is built as a Retrieval-Augmented Generation (RAG) pipeline. A user question flows through five stages, from ingesting source documents to generating a grounded answer:

```mermaid
flowchart TD
    A["📄 Document Ingestion<br/>Python · local text/HTML/Markdown docs"]
    B["✂️ Chunking<br/>custom chunk_text()<br/>500-char chunks · 100-char overlap"]
    C["🧮 Embedding + Vector Store<br/>sentence-transformers (all-MiniLM-L6-v2)<br/>stored in ChromaDB"]
    D["🔍 Retrieval<br/>semantic search · top-k = 4"]
    E["💬 Generation<br/>LLM answers using retrieved chunks + source metadata"]

    A --> B --> C --> D --> E
```

---

## AI Tool Plan

1. I plan to use Claude as a brainstorming and feedback tool by giving it my planning sections and asking it to point out unclear parts or weaknesses. I will compare its suggestions with the assignment requirements before deciding what to change.

2. I plan to use Claude to refine my writing by giving it rough drafts of my Domain, Chunking Strategy, Retrieval Approach, and Evaluation Plan sections. I will make sure the revised wording still reflects my own project choices.

3. I plan to use Claude to help implement my `chunk_text()` function by giving it my chunk size, overlap size, and document types. I will test the function myself to check that chunks are readable and keep important context together.

4. I plan to use Claude to review retrieval results by giving it a test question, the retrieved chunks, and my retrieval settings. I will use the conversation to reason through whether the results are relevant instead of accepting its feedback automatically.

5. I plan to use Claude to improve my evaluation questions by asking whether each question is specific and testable. I will verify that each final question can be answered using my collected documents.

6. Overall, I will use Claude thoughtfully as a support tool, not as a replacement for my own decisions. I will combine its feedback with my own judgment and the project instructions before submitting anything.

**Milestone 3 — Ingestion and chunking:** I gave Claude my Chunking Strategy section and document types and asked it to implement `chunk_text()`. It produced a sliding-window chunker (500 characters, 100 overlap) that snaps chunk boundaries to paragraph/sentence breaks, plus a cleaning step (HTML/entity/boilerplate removal) and an inspection step with warnings. I verified the output by reading representative and random chunks before accepting the parameters.

**Milestone 4 — Embedding and retrieval:** I used `all-MiniLM-L6-v2` with ChromaDB and top-k = 4. Claude flagged that ChromaDB defaults to L2 distance, which would not match my planned 0.5/0.6/0.7 relevance thresholds, so I configured cosine distance instead. I tested retrieval on the evaluation questions and confirmed relevant chunks ranked first at low distances.

**Milestone 5 — Generation and interface:** I connected retrieval to Groq `llama-3.3-70b-versatile` with a strict grounding prompt and built a Gradio UI. Adding Gradio 6.x forced an upgrade of the embedding stack (sentence-transformers/transformers); I verified retrieval distances were unchanged before keeping it. I tested grounding with an off-topic question and confirmed the system refuses instead of hallucinating.
