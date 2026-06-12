"""
Unofficial Guide to Ordering Matcha — Milestone 5 (part 2): Gradio Interface

A simple web UI on top of ask() from query.py. The user types a question,
presses Ask (or Enter), and sees the grounded answer plus the source files the
answer was drawn from.

Run it with:
    python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import gradio as gr

from query import ask


def handle_query(question):
    """
    Called when the user submits a question. It runs the grounded-generation
    pipeline and returns two strings: the answer, and a bulleted source list.
    """
    # Guard against an empty submission so we don't waste an API call.
    if not question or not question.strip():
        return "Please type a question first.", ""

    result = ask(question)

    # Build a readable bullet list of sources. If there are none (e.g. the model
    # refused because the context was insufficient), say so clearly.
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — answer was not grounded in the documents)"

    return result["answer"], sources


# Build the interface.
with gr.Blocks(title="Unofficial Guide to Ordering Matcha") as demo:
    gr.Markdown("# 🍵 Unofficial Guide to Ordering Matcha\n"
                "Ask a beginner question about ordering matcha. Answers come "
                "only from the project's documents.")

    inp = gr.Textbox(label="Your question",
                     placeholder="e.g. What matcha drink is best for a beginner?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    # Clicking the button submits the question.
    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    # Pressing Enter in the textbox also submits.
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
