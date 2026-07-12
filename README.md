# When Does Reranking Pay Off? — RAG Chunking/Embedding/Reranking Study

Reproducible study of chunk size, embedding model, and cross-encoder reranking trade-offs in RAG, on SQuAD v2 (single-hop) and HotpotQA (multi-hop).

## Key finding
Reranking (BAAI/bge-reranker-base) showed **no statistically significant effect** on retrieval quality on either dataset (paired Wilcoxon, p=0.50 SQuAD / p=0.66 HotpotQA), while chunk size was the dominant factor on HotpotQA (Hit@k roughly doubled from 256→1024 tokens). Full writeup: [`papers/`](papers/).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --break-system-packages
```

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` (free, no card needed — console.groq.com).

## Reproduce

```powershell
python src/load_data.py              # download + subsample SQuAD v2 / HotpotQA (seed=42)
python src/squad_experiments.py      # full sweep on SQuAD v2 (~1.5-2 hrs on Groq free tier)
python src/hotpot_experiments.py     # full sweep on HotpotQA (resumable across daily quota resets)
python src/analysis.py               # tables, significance tests, figures -> results/
```

`squad_experiments.py` supports `--smoke-test` for a ~1-minute sanity check before the full sweep.

## Scope notes (see paper Limitations for full disclosure)
- Embedding sweep uses only BAAI/bge-small-en-v1.5 (free) — the originally planned OpenAI text-embedding-3-small comparison was cut due to no billing being configured; this is a disclosed scope-control decision, not an oversight.
- Groq's free-tier daily token cap (~500k tokens/model/day) means a full two-dataset sweep does not fit in one calendar day; `hotpot_experiments.py` checkpoints and skips already-completed configs so it can be resumed across days.

## Repo structure
```
config.yaml              # experiment design: chunk sizes, embeddings, reranking, seeds
src/
  load_data.py            # dataset download + subsampling
  pipeline.py             # embeddings, reranker, splitter, retrieval metrics, caching
  squad_experiments.py    # SQuAD sweep runner
  hotpot_experiments.py   # HotpotQA sweep runner (resumable)
  analysis.py             # Day 5: tables, significance tests, figures
data/                     # cached subsampled datasets (committed for exact reproducibility)
results/                  # raw results CSVs, main_effects_table.csv, decision_rule.txt, figures/
papers/                   # paper draft (Markdown) + final IEEE-format PDF
cache/                    # per-question LLM response cache (gitignored)
```

## Notes
- Reranker: switched from `FlagEmbedding.FlagReranker` to `sentence-transformers.CrossEncoder` after hitting a `prepare_for_model` compatibility error with modern `transformers` versions.
- Generator: switched from Gemini to Groq (`llama-3.1-8b-instant`) after hitting Gemini's free-tier `limit: 0` issue.