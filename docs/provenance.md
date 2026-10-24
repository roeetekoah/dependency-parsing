# Editions, credit, and later changes

The selected source is the working NLP exercise 4 directory. A local audit compared the submission ZIP to that directory: all 12 Python/README members matched byte-for-byte. The report and decoder appendix are separate artifacts; no repository history or person-specific change log was available. A pre-extraction SHA-256 inventory of all 323 source files is kept in the private portfolio assessment, outside this release.

## Attribution supported by the record

| Component | What can be established |
|---|---|
| Parser, corrected decoder, experiment scripts, report | Source README/report credit Roee Tekoah and Yona Gan jointly. No verified person-by-person split. |
| Baseline problem, split, word/POS features, two passes, learning rate 1 | Prescribed by the assignment; not claimed as original research choices. |
| `chu_liu_edmonds.py` | Assignment explicitly supplied this decoder; source README says it was left unmodified. No independently obtained pristine handout was available to verify that claim. |
| `transformer_parser_utils.py` | Assignment supplied the extractor; source README says it was unmodified. Same pristine-handout limitation. |
| Corrected cycle reconstruction | Jointly credited coursework correction, developed from the supplied decoder; not assigned solely to Roee. |
| Per-head sweep and learned combiner | Present as beyond-assignment coursework extensions; jointly credited pending individual clarification. |
| Corpus, pretrained BERT, parsing algorithm | Third-party data/model and established algorithmic work, not authored here. |
| This extraction and strengthening | Later AI-assisted local release work on 11 October 2026; not retroactively credited as original coursework. |

## Source-to-release mapping

| Source | Release destination/change |
|---|---|
| `mst_parser.py` | `perceptron.py`: feature/update/lazy-average logic retained; local RNG, argument validation, package imports. Loading moved to `data.py`; obsolete exercise entrypoint removed. |
| `chu_liu_edmonds_fixed.py` | `decoder.py`: contraction/expansion endpoint correction retained; new validation, deterministic ties, one incoming-edge scan, linear cycle walk, explicit stack rather than recursion, explicit infeasibility exception. |
| Supplied decoder | Excluded. The minimal defect is explained with a numeric fixture; no unsafe legacy option in the public API. |
| Supplied extractor | Excluded. `attention.py` implements the documented aggregation through the authored per-head vectorization; records its origin and verifies against an independent explicit loop. |
| `attention_parser.py`, `bonus_heads.py`, `bonus_combine.py` | `attention.py` and `cli.py`: consolidated mean-head, train-selected head, test oracle, and learned combiner. CPU/eager backend, pinned future model revision, explicit downloads, alignment/length failures. |
| `robustness.py`, `collect_results.py` | Configured CLI and JSON output; supplied-decoder experiments retained only as historical records, no monkey-patching. |
| `download_data.py` | Excluded; separate documented corpus acquisition and explicit model-download flag. |
| `results.txt`, report tables | Normalized CSV, evidence labels, metric/split notes. Originals are preserved outside the release. |
| Original report and primers | Excluded from publication candidate: coursework/scaffolding rights and private metadata not silently propagated. This release supplies a focused new technical explanation. |
| Profile's conceptual arc sketch | Rebuilt in a related green palette as release-owned light/dark SVGs with an explicit conceptual caption. No profile files are changed. |

The release is the canonical standalone extraction; the historical working directory remains authoritative evidence of the submitted edition. Neither a clean package nor a passing test suite proves personal authorship, redistribution permission, empirical generalization, or user acceptance.
