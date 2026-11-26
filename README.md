# ResearchPilot

A multi-agent research assistant. Give it a topic and it plans an outline, waits for you to approve it,
searches and reads the web, writes a cited report, and removes any sentence its sources do not back.
You can then ask for changes (only the affected sections are redone), run a sentiment and recommendations
analysis, and chat with the report through a RAG chatbot.

Built with LangGraph, LangChain, FastAPI, Chroma, fastembed and LangSmith.

- Architecture and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Evaluation results and the v1 vs v2 comparison: [docs/EVALUATION.md](docs/EVALUATION.md)

## Setup

```bash
./start.sh
```

That one command installs `uv` if it is missing, installs the dependencies, checks that `.env` exists and that the
API keys in it work (LLM, fallback LLM, search, LangSmith, Brevo), starts the server on the first free port from 8000
and opens the browser. On a fresh checkout it creates `.env` from `.env.example` and stops so you can fill in the keys.

By hand it is:

```bash
brew install uv
uv sync
cp .env.example .env      # then fill in the keys you have
uv run python scripts/check_setup.py
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. No Docker is needed: checkpoints and accounts live in SQLite and vectors in Chroma, all under `data/`.

## Accounts

With `AUTH_ENABLED=true` visitors only see the welcome page. Everything else needs an account:
sign up, enter the 6-digit code sent by e-mail (Brevo), and you are in. "Forgot it?" sends a one-time reset link.
Reports and chats belong to the user who created them. Without `BREVO_API_KEY` the code is printed in the
server log instead, which is enough for local testing.

## Environment variables

| Variable | What it does |
|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | Main model. Provider is `groq`, `google`, `anthropic` or `openai`. Change these two lines to switch. |
| `LLM_FALLBACK_PROVIDER`, `LLM_FALLBACK_MODEL` | Takes over for a call when the main provider errors. Default `groq`. |
| `LLM_FAST_MODEL` | Smaller model for the claim judge and the edit planner. Must be a model of `LLM_PROVIDER`. Empty means use the main model. |
| `GROQ_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Only the keys for the providers you use. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Tracing. Every run is tagged with the report id. |
| `EMBEDDING_MODEL` | `fastembed:BAAI/bge-small-en-v1.5` (local, no key), `hash`, or `openai:<model>`. |
| `RETRIEVAL_MODE`, `TOP_K`, `MIN_SCORE` | `dense`, `bm25` or `hybrid`. Chunks under `MIN_SCORE` are ignored. |
| `SEARCH_PROVIDER` | `tavily` (needs `TAVILY_API_KEY`, fastest and fine with parallel calls), `firecrawl` (needs `FIRECRAWL_API_KEY`), `ddgs` (no key) or `wikipedia`. The others act as backups. |
| `RESEARCH_MAX_SECTIONS`, `RESEARCH_SOURCES_PER_SECTION`, `RESEARCH_PARALLEL_SECTIONS` | Size and speed of a report. |
| `RESEARCH_VERIFY` | `false` skips claim verification. Used as the baseline in the evals. |
| `RESEARCH_MAX_LLM_CALLS` | Hard cap on LLM calls per run. |
| `AUTH_ENABLED` | `true` puts everything except the welcome page behind sign-in. |
| `BREVO_API_KEY`, `BREVO_SENDER_EMAIL`, `BREVO_SENDER_NAME` | Sends the sign-up code and reset link. The sender is picked from the Brevo account when left empty. |
| `SESSION_TTL_DAYS` | How long a login lasts. |

## Using it

1. Sign in, then type a topic on the new-report screen. The agent plans an outline and stops.
2. Edit the outline if you want (rename, reorder, add, remove) and approve it.
3. Watch the sections being searched, read, written and checked. Click any citation to see the passage behind it.
4. Ask for changes in the box at the bottom, or use the buttons on a section (More detail, Better sources, Rewrite, Remove).
5. Open the Analysis tab for sentiment and recommendations.
6. Open the Chat tab, add the report to the knowledge base and ask questions.
7. The Activity tab shows what each revision changed and what each run cost in time, tokens and calls.

## Tests and evals

```bash
uv run pytest                      # 91 tests, no network or API key needed
uv run pytest -m contract          # the original milestone contract tests
uv run ruff check

uv run python -m evals.run --label v1 --no-verify --retrieval dense   # baseline
uv run python -m evals.run --label v2                                 # full system
uv run python -m evals.compare v1 v2
```

## Layout

```
app/
  research/
    graph.py       main LangGraph graph: guard, plan, outline review, sections, assemble, feedback loop
    section.py     per-section worker graph: search, read, write, verify
    verify.py      claim parsing and the two-stage support check
    analysis.py    analysis graph: sentiment and recommendations
    kb.py          report to vector store, section by section
    search.py      search providers (swappable)      reader.py   safe page fetching
    service.py     background runs, report snapshots  store.py    one JSON file per report
    usage.py       token, call and cost tracking      trust.py    trust scores
  auth/            accounts: sign-up code, sign-in, reset link, SQLite store, Brevo e-mail
  chat/            RAG chat engine (routing, guardrails, memory, tool approval)
  vectordb/        chunking, embeddings, Chroma store, dense / BM25 / hybrid retriever
  api/             JSON endpoints
evals/             test topics, judges, metrics, runner, comparison
templates/, static/   the UI: welcome page (three.js + Motion + the scroll engine), auth page, app pages
tests/
```
