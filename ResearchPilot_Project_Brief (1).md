# ResearchPilot: A Multi-Agent Research & Insights Platform

## 1. Overview

You will build **ResearchPilot**, an AI system that works like a junior research analyst. A user gives it a topic, and it goes off, searches the web, reads the sources, and comes back with a well-structured **research report with citations**. The user can then steer the work ("expand section 3", "add a subtopic on X"), get a **sentiment analysis and recommendations**, and finally **chat with the report** through a knowledge-base chatbot. Throughout, an **evaluation system** measures how good the output actually is.

This project brings together everything from the training: LLM apps, RAG, LangChain, LangGraph, agents, multi-agent systems, human-in-the-loop (HITL), and LangSmith observability.

## 2. The Problem

Doing research manually is slow: you search, open many tabs, skim, take notes, write, and then cite. LLMs can speed this up, but a plain chatbot answer has three big problems:

- It may **hallucinate** facts.
- It usually gives **no verifiable sources**.
- The user has **no control** over the process.

Your job is to build a system that is **trustworthy (grounded and cited), controllable (human can steer it), and measurable (we can prove how good it is)**.

## 3. What the System Must Do

### Module 1: Research & Report Agent

**Input:** a topic (e.g., "State of electric two-wheelers in India").

**Output:** a compiled report with sections, inline citations, and a reference list.

Expected behavior:

1. **Plan:** break the topic into an outline of sections and subtopics.
2. **Search:** find relevant sources on the web for each section.
3. **Read:** fetch the pages and extract the relevant content.
4. **Write:** draft each section using *only* the gathered content.
5. **Cite:** every factual claim must link to a source. Include a final reference list.
6. **Verify:** check that cited sources actually exist and support the claims.

**Human-in-the-loop requirements:**

- The user can **review and approve (or edit) the outline** before the full report is written.
- After the report is generated, the user can ask for changes such as:
  - "Add a subtopic on government policy."
  - "Make section 3 more detailed."
  - "Replace the weak sources in section 2."
- The agent must **update only the affected section(s)**, not regenerate the entire report from scratch.

### Module 2: Analysis Agent (Sentiment & Recommendations)

Takes the finished report (and/or the sources behind it) and produces:

- **Sentiment analysis:** overall sentiment on the topic, plus sentiment per section and/or per source (positive / neutral / negative, with a short justification).
- **Recommendations:** key takeaways, opportunities, risks, and suggested next steps, all traceable back to the report content.

This can be a separate agent/graph or part of the same one. Justify your design choice in your documentation.

### Module 3: Knowledge Base & Chatbot

- The user can **ingest the report** into a knowledge base (chunk, embed, store in a vector database).
- A **RAG chatbot** lets the user ask questions about the report.
- Answers must be **grounded in the report** and show **which source/section** they came from.
- If the answer is not in the knowledge base, the bot should say so instead of making something up.
- The chatbot should handle **follow-up questions** (conversation memory).

### Module 4: Evaluation & Observability

You must prove your system works, not just claim it. Build an evaluation setup that scores your system on a **fixed set of test topics** (at least 10 to 15).

**Report quality**

| Metric | What it checks |
|---|---|
| Citation validity | Do cited URLs exist, and do they actually support the claim? |
| Groundedness | What percentage of claims are supported by retrieved content (hallucination rate)? |
| Coverage | Does the report cover the important aspects of the topic? |
| Source diversity | Are sources varied, or does the report lean on one site? |
| Structure & readability | Clarity, organization, and flow (LLM-as-judge with a rubric) |

**Chatbot quality**

| Metric | What it checks |
|---|---|
| Faithfulness | Is the answer supported by retrieved context? |
| Answer relevancy | Does it actually answer the question? |
| Context precision / recall | Did retrieval fetch the right chunks? |
| "I don't know" handling | Does it refuse correctly when the answer isn't in the documents? |

**System metrics**

- Latency per run
- Token usage / estimated cost
- Number of tool and search calls

Use **LangSmith** to trace every run so you can debug and compare versions.

## 4. High-Level Architecture

```
                    ┌─────────────────────────────┐
                    │            UI               │
                    │  (topic input, report view, │
                    │   feedback, chat window)    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │        Backend / API        │
                    └───┬─────────┬─────────┬─────┘
                        │         │         │
         ┌──────────────▼───┐ ┌───▼──────┐ ┌▼───────────────┐
         │ Research Agent   │ │ Analysis │ │ Knowledge Base │
         │ (LangGraph)      │ │ Agent    │ │ + RAG Chatbot  │
         │ plan → search →  │ │ sentiment│ │ chunk → embed →│
         │ read → write →   │ │ + recs   │ │ retrieve → ans │
         │ verify  (+HITL)  │ │          │ │                │
         └────────┬─────────┘ └────┬─────┘ └───────┬────────┘
                  │                │               │
                  └────────────────▼───────────────┘
                     LangSmith tracing + Evaluation
```

## 5. Deliverables

1. **Working application** (UI + backend) demonstrating all four modules.
2. **Source code** in a Git repository with a clear README (setup steps, how to run, environment variables).
3. **Architecture document:** diagrams of your agent graphs, state design, and the reasoning behind key decisions.
4. **Evaluation report:** results on your test topics, and a **comparison of at least two versions** of your system (for example, different prompts, models, retrieval settings, or with/without a verification step). Explain what changed and what you learned.
5. **LangSmith traces/dashboard** (screenshots or shared links) showing a few representative runs.
6. **Final demo and presentation** for the whole group.

## 6. Ground Rules

- **No citation, no claim.** Every factual statement in a report must trace back to a source.
- **Do not fabricate sources.** A made-up URL or reference is a serious failure.
- **Every group member** should contribute to code and understand the whole system.
- Keep API keys out of your repository (use `.env` files and `.gitignore`).
- Free-tier APIs have rate limits. Design your code so the **LLM and search providers can be swapped** easily.
- Use of AI coding assistants is fine, but you must be able to explain every part of your code.

## 7. Recommended Tech Stack (Free / Open-Source)

These are suggestions to get you started. **You are free to use other tools if you are more comfortable with them**, as long as you meet the requirements above and can justify your choices.

| Layer | Recommended options |
|---|---|
| Orchestration | LangChain + LangGraph |
| LLM | Gemini API free tier, Groq free tier (Llama / Qwen models), or Ollama (local models) |
| Web search | DuckDuckGo via the `ddgs` package (no API key), Tavily free tier, or self-hosted SearXNG |
| Extra sources | Wikipedia API, arXiv API, Semantic Scholar API, free news APIs (NewsAPI / GNews) |
| Page content extraction | `trafilatura`, `newspaper3k`, `BeautifulSoup` |
| Embeddings | `sentence-transformers` (e.g., BGE-small, all-MiniLM) or Ollama embeddings |
| Vector database | Chroma (simplest), FAISS, or Qdrant |
| Sentiment analysis | HuggingFace models (e.g., `cardiffnlp/twitter-roberta-base-sentiment`, FinBERT for finance) and/or LLM-based analysis |
| Evaluation | LangSmith evals, RAGAS, DeepEval |
| Observability | LangSmith (free developer tier) |
| Backend | FastAPI |
| UI | Streamlit or Gradio (fastest), or React if you prefer |
| Persistence / checkpoints | SQLite or Postgres (LangGraph checkpointer) |

> Free tiers and rate limits change often. Verify the current limits of any service before you build around it.

## 8. Suggested Design Hints

- **Shared state:** design a clear LangGraph state (topic, outline, sections, sources, citations, feedback) so sections can be updated independently.
- **Section-level editing:** store each section separately with its own sources so a change request only re-runs the relevant part.
- **Checkpointing and `interrupt()`:** these are the natural way to implement the outline approval and feedback steps.
- **Citations:** keep source IDs attached to retrieved text from the start, rather than adding citations afterward.
- **Chunk metadata:** store section name and source URL with every chunk so the chatbot can cite where an answer came from.
- **Start simple:** get one end-to-end path working first, then improve each part.

## 9. Optional Stretch Goals

Pick these only after the core is solid:

- Export the report to PDF or DOCX
- A dedicated fact-checker agent that re-verifies claims
- Multi-language report generation
- A cost-aware router that uses a cheap model for easy tasks and a stronger model for hard ones
- Report version history with diffs between revisions
- Comparing two topics side by side

## 10. How You Will Be Evaluated

| Area | What we look for |
|---|---|
| Functionality | All four modules work end to end |
| Agent design | Sensible graph/state design, correct use of HITL and checkpointing |
| Trustworthiness | Real citations, low hallucination, correct "I don't know" behavior |
| Evaluation rigor | Meaningful metrics, a proper test set, a thoughtful version comparison |
| Code quality | Clean structure, readable code, setup documentation |
| Presentation | Clear demo, clear explanation of decisions and trade-offs |

Good luck, and build something you would trust to do your own research.
