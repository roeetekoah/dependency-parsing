# Rights and excluded material

No general redistribution license is granted by this repository. It contains jointly credited coursework and an adaptation of a corrected decoder that developed from supplied assignment code. The local records do not establish collaborator consent or the instructor's redistribution terms. Retaining credit alone does not establish permission.

| Material | Boundary |
|---|---|
| Coursework implementation/report | Jointly credited to Roee Tekoah and Yona Gan; individual allocation unconfirmed |
| Original assignment, supplied decoder/extractor | Not shipped; course terms absent from inspected files; decoder lineage remains disclosed |
| NLTK dependency treebank | Not shipped, including fixture sentences; obtain separately under its own terms |
| BERT weights/tokenizer | Not shipped; optional external download pinned for future runs |
| New synthetic fixtures and conceptual SVGs | Authored for this release; not corpus/model output |
| New tests/interface/documentation | AI-assisted release additions, distinguished from coursework |

NLTK's official [dataset-license summary](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md) lists the dependency treebank as a Penn Treebank fragment with non-commercial/no-redistribution terms. Do not infer corpus rights from NLTK's software license. The configured BERT model's [official model card](https://huggingface.co/google-bert/bert-base-uncased/blob/b84905490c2d4796d85a42590c88a95af94cd5e2/README.md) declares Apache-2.0; that does not license this coursework or redistribute the underlying pretraining texts. Sources checked 11 October 2026.

Optional dependencies remain external: NLTK, NumPy, PyTorch, Transformers, scikit-learn, pytest, setuptools, and wheel. Installation does not make their licenses interchangeable. No third-party source or model binaries are vendored here.
