# Evaluation report

## Setup

- **Test set:** 12 fixed topics in `evals/topics.json`. Each topic lists four aspects a good report should cover.
- **Report size:** 3 sections per report (to fit the free-tier rate limit of 8,000 tokens a minute).
- **Writer model:** `openai/gpt-oss-120b` on Groq. **Judge model:** `qwen/qwen3.8-27b`, a different model family,
  so the system does not grade its own work.
- **Same evidence for both versions:** searches and fetched pages are cached on disk, so v1 and v2 read the same pages.
- **Chatbot questions:** for each report, the planner's own questions for the first two written sections (answerable)
  and two unrelated questions (must be refused).

Run it:

```bash
uv run python -m evals.run --label v1 --no-verify --retrieval dense
uv run python -m evals.run --label v2
uv run python -m evals.compare v1 v2
```

The runner skips topics that are already in the result file, so it can be stopped and continued.

## The two versions

| | v1 (baseline) | v2 (full system) |
|---|---|---|
| Claim verification | off: every sentence the writer produces is kept | on: local check, LLM judge for unsure sentences, one rewrite |
| Chat retrieval | dense | hybrid (dense + BM25, reciprocal rank fusion) |

## How each metric is computed

| Metric | Method |
|---|---|
| Citation validity: URL resolves | Live HTTP request to every cited URL. |
| Citation validity: source supports the claim | Judge sees each cited sentence with the passages of the sources it cites. Share judged supported. |
| Groundedness | Same judgement over all sentences. A sentence with no citation counts as not grounded. Hallucination rate is 1 minus this. |
| Coverage | Judge checks each of the topic's four expected aspects against the report text. |
| Source diversity | Unique domains divided by cited sources. |
| Structure and readability | Judge scores clarity, organisation and flow from 1 to 5. Reported as the total out of 15, scaled to 0..1. |
| Faithfulness, answer relevancy | Judge sees question, answer and the retrieved sections. |
| Context recall | 1 if any retrieved chunk comes from the section the question was planned for. |
| Context precision | Share of retrieved chunks that come from that section. This is strict: a chunk from a neighbouring section that also answers the question counts as a miss. |
| "I don't know" handling | Out-of-scope questions must get a refusal. |
| Latency, tokens, calls, cost | Measured by a callback during the run. Cost uses list prices. |

## Results

**Status: interim.** At the time of writing, 3 of the 12 topics had finished in both versions (the free tier allows
about one topic every eight minutes for the pair). The run was still going. Re-run `evals.compare` for the final table.

| Metric | v1 | v2 |
|---|---:|---:|
| Citation validity: URL resolves | 1.00 | 1.00 |
| Citation validity: source supports the claim | 0.62 | 0.74 |
| Groundedness (supported sentences) | 0.55 | 0.74 |
| Hallucination rate (lower is better) | 0.45 | 0.26 |
| Uncited sentences per report (lower is better) | 2.33 | 0.00 |
| Coverage of expected aspects | 0.92 | 0.92 |
| Source diversity (unique domains / sources) | 0.97 | 0.92 |
| Structure and readability (rubric) | 0.80 | 0.87 |
| Sentences per report | 22 | 18 |
| Chat faithfulness | 0.83 | 1.00 |
| Chat answer relevancy | 1.00 | 0.83 |
| Context precision | 0.38 | 0.42 |
| Context recall | 1.00 | 1.00 |
| Correct refusals (out-of-scope questions) | 1.00 | 1.00 |
| Latency per report (s) | 41 | 86 |
| LLM calls per report | 4.0 | 8.3 |
| Input tokens per report | 3,960 | 7,845 |
| Output tokens per report | 1,243 | 1,924 |
| Estimated cost per report (USD) | 0.0013 | 0.0019 |
| Search calls per report | 6 | 6 |
| Pages read per report | 11 | 10 |

Topics in this table: electric two-wheelers in India, solid-state batteries, remote work and productivity.

## What changed and what we learned

1. **Verification works, and it costs time.** Groundedness went from 0.55 to 0.74 and uncited sentences from 2.3 per
   report to zero. The price is about twice the latency and tokens, because rejected drafts get one rewrite.
2. **Coverage did not drop.** Removing 4 to 6 sentences per report did not remove whole aspects (0.92 in both).
   Reports got shorter (22 to 18 sentences) and scored slightly higher on readability.
3. **Verification is not finished.** An independent judge still rejects about a quarter of v2's sentences. Reading them,
   they are mostly long sentences that join two facts where the source backs only one. Next step: send every sentence
   to the judge, not just the unsure ones, or split compound sentences before checking.
4. **Without the verifier the writer breaks the rules.** Even with a prompt that demands a citation on every sentence,
   v1 left 2.3 sentences per report uncited. Prompting alone is not a guardrail.
5. **Refusals are reliable.** Both versions refused every out-of-scope question, thanks to the similarity threshold.
6. **Chat numbers are too small to rank the versions.** Six answerable questions per version is not enough to say
   hybrid beats dense. Context precision is low by construction (see the table above): sections of one report overlap.
7. **A bug the eval found.** The first eval run showed two empty sections per report. The cause was the model writing
   citations as `【1】` instead of `[1]`, so the parser treated every sentence as uncited. Fixed and covered by a test.

## LangSmith

Every run is traced to the project named in `LANGSMITH_PROJECT`.

- Filter by tag `researchpilot` for product runs, `eval` plus `v1` or `v2` for eval runs.
- Research runs carry `report_id` in metadata, chat runs carry `session_id`.
- Useful traces to screenshot: one `research_research` run (shows the parallel `section` fan-out with
  `section_search`, `section_read`, `section_write`, `section_verify`), one `research_revision` run (shows a single
  section re-running), one `analysis` run, and one `chat_turn` with a retriever run inside it.
