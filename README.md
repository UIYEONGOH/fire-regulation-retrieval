# Retrieval for AI-Assisted Review of Korean Building Fire Safety Regulations

Database, benchmark, code, and results for the paper
"Retrieval Benchimark for AI-Assisted Review of Korean Building Fire Safety Regulations" (IEEE Access, under review).

## Contents

```
data/
  lawdb_v2_2_frozen.sqlite   integrated regulation database (111 regulations, 2,719 retrieval units)
  lawdb_v2_1_frozen.sqlite   previous version, used only for the citation-graph sensitivity runs
  goldset_v3.csv             benchmark: 543 queries, 785 query-provision pairs
  goldset_v3_manual_corrections.csv   the nine gold sets corrected by hand, with reasons
  delegation_candidates.csv  the 727 citations reclassified from reference to delegation (REF_CANDIDATE.ref_id)
  cache/                     BGE-M3 dense embeddings of the retrieval units and queries
results/                     outputs of the runs reported in the paper
PREREG_v1.0.md               preregistered analysis plan (Korean), sha256 537c7cad...efd54
PREREG_v1.0_implementation_notes.md   implementation details fixed before the runs (Korean)
common.py                    data loading, BM25, dense scoring, metrics, cross-validation, graph scoring
nn.py                        BGE-M3 query encoder and bge-reranker-v2-m3
s1_fast.py, s5_rerank.py, s6_latency.py, s7_report.py   experiment steps
```

## Database

SQLite. Main tables:

| table | rows | content |
|---|---|---|
| LAW | 111 | regulation metadata |
| PROVISION | 15,300 | article, paragraph, subparagraph, and item text |
| APPENDIX / APPENDIX_LINE | 189 / 19,330 | attached tables and their lines |
| REF_CANDIDATE | 5,949 | citations between provisions with type and resolution |
| APPENDIX_REF | 1,340 | citations from attached tables |

The texts were collected through the Open API of the Korean Law Information Center (open.law.go.kr) as the versions in force on November 1, 2026.

## Benchmark

`goldset_v3.csv`, one row per query.

| column | content |
|---|---|
| cid | case id (edition-number) |
| edition | casebook edition: 2024, 2025, 2026 |
| question | inquiry text |
| gold_units | retrieval units cited as grounds in the official reply, separated by `; ` |
| systems, gold_kinds, cross_system, question_has_cue | subset labels used in RQ2 |

Queries come from the Inquiry-Reply Casebook on Fire Facility Laws published by the National Fire Agency.

## Conditions

| id | method |
|---|---|
| C1 | BM25, Kiwi morphological tokens, k1 = 0.9, b = 0.4 |
| C2 | BM25, jamo 3-grams, k1 = 0.9, b = 0.4 |
| C3 | BGE-M3 dense, 2,000-character chunks, MaxP |
| C4 | C3 + Rocchio, k = 15, alpha = 0.3 |
| C5 | RRF(C3, C1), k = 60 |
| C6 | SAR citation-graph scoring over C3, k = 15, beta = 0.3 |
| C7 | bge-reranker-v2-m3 over the top 30 of C5 |
| G1, G2 | scope-weighted graph scoring and in-degree prior, selected by grouped cross-validation |

## Running

Tested with Python 3.14 on Windows 11, CPU only.

```
pip install -r requirements.txt
python s1_fast.py       # C1-C6, G1, G2, tuning, sensitivity; no model download
python s5_rerank.py     # C7; downloads bge-reranker-v2-m3; about 16 h on a CPU
python s6_latency.py    # latency; downloads BGE-M3
python s7_report.py     # tables, statistics, manifest -> results/report/
```

Each script verifies the sha256 of the data files and of `PREREG_v1.0.md` before running. Results are written to `results/`.
`s5_rerank.py` saves progress per query and resumes if interrupted.

## Results

`results/` holds the outputs of the runs reported in the paper (AMD Ryzen 7 8840U, 16 GB RAM, CPU only; see `env` in `results/runs/C7.json`).

| file | content |
|---|---|
| runs/*.npz, runs/*.json | top-100 ranking and scores per query for every run |
| work/rerank_C7.jsonl | reranker scores and time per query for C7 |
| s1_selection.json | configurations selected in each cross-validation fold |
| s6_latency.json | per-query component timings |

`python s7_report.py` builds `results/report/` from these files without running any model:

| file | content |
|---|---|
| table_all_conditions.csv | mean metrics per condition |
| main_comparisons.csv | six preregistered comparisons, query and cluster bootstrap, Holm-adjusted p |
| descriptive_comparisons.csv | other comparisons, intervals only |
| subsets.csv | R@10 and nDCG@10 by query subset |
| latency.csv | median and p95 query latency |
| per_query_metrics.csv.gz | per-query metrics for every run |
| rankings/*.csv.gz | rankings as CSV |
| manifest.json | hashes of inputs and scripts, environment |

## Models

| model | revision |
|---|---|
| BAAI/bge-m3 | as downloaded (cache embeddings included in `data/cache`) |
| BAAI/bge-reranker-v2-m3 | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |

## License

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`LICENSE-DATA`).

Sources:
- Regulation texts: Korean Law Information Center (open.law.go.kr), Ministry of Government Legislation. Statutes and public notices are not protected works under Article 7 of the Korean Copyright Act.
- Queries: Inquiry-Reply Casebook on Fire Facility Laws, National Fire Agency (2024, 2025, and 2026 editions), KOGL Type 1 (attribution).
