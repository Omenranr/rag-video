#!/usr/bin/env python3
"""
Gradio UI for video → (transcript + detections) → hybrid search → chat (multi-LLM)
+ Providers: OpenAI • Anthropic
+ Optional Web Search with Exa
+ Clickable timecodes to seek the video

NEW (2025-09-07):
- Answers now use **RAG + chat history**.
- We (1) rewrite the latest user message into a **standalone question** using recent chat history,
  (2) retrieve RAG context with that question,
  (3) route to web search if needed,
  (4) answer with the **conversation window + RAG + (optional) web snippets**.

Flow:
1) Contextualize the latest user message with chat history → standalone question.
2) Retrieve transcript context (hybrid + rerank + ±N neighbors) with that standalone question.
3) Ask the chosen LLM to decide if web search is needed:
   - If NO: LLM returns final answer (uses transcript context + chat history).
   - If YES: LLM returns an EXACT search query string.
4) If search needed: call Exa API → fetch results → second LLM call
   to answer using transcript context + web snippets + chat history.

Also:
- Reuse existing outputs per video (outputs/<video_stem>_timestamp).
- Clickable timecodes list (radio) that seeks the video player.

Run:
  pip install gradio==4.* pandas numpy tqdm rank-bm25 sentence-transformers requests python-dotenv
  python app_multillm.py

Notes:
- Choose your LLM provider in the UI. Only the credentials for the selected provider are used.
- For OpenAI, you may override the Base URL to use compatible endpoints; leave blank for api.openai.com.
- For Anthropic, the standard Messages API endpoint is used.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from ui.layout import demo

load_dotenv()

if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        share=False,
        debug=True
    )
