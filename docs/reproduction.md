# Reproduction routes

Run commands from the repository root. Python 3.11+ is supported by packaging; the checked local environment was Python 3.13.13. Dependency ranges in `pyproject.toml` are compatibility ranges, not an exact historical lock. `experiments/verified-environment.txt` records versions checked locally; the original coursework environment was not recovered.

## Small offline route

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install .
python -m dependency_parsing demo
```

On macOS/Linux activate with `source .venv/bin/activate`. The wheel has zero required third-party dependencies, and the demo never loads NLTK or Transformers. It trains and evaluates on the same synthetic examples and is explicitly not an accuracy benchmark.

For independent decoder and representation tests:

```text
python -m pip install ".[test]"
python -m pytest -q
```

## Classical experiment

Obtain NLTK's dependency treebank separately under its own terms; it is not bundled. The downloader requires network access. Keep the cache out of version control.

```text
python -m pip install ".[corpus]"
python -m nltk.downloader -d .cache/nltk_data dependency_treebank
python -m dependency_parsing run --config experiments/classical.json --data-dir .cache/nltk_data --output runs/classical.json
python -m dependency_parsing run --config experiments/robustness.json --data-dir .cache/nltk_data --output runs/robustness.json
```

An existing NLTK data directory can replace `.cache/nltk_data`; it must contain `corpora/dependency_treebank/*.dp`. The loader reads this explicit directory, without silently falling back to another cache. It preserves file and sentence order. Runtime depends on sentence length and contraction count; consult the checked run record rather than the old README's hardware-unspecified minute estimate.

## Frozen-attention experiments

```text
python -m pip install ".[corpus,attention]"
python -m dependency_parsing run --config experiments/attention.json --data-dir .cache/nltk_data --model-cache .cache/models --allow-download --output runs/attention.json
python -m dependency_parsing run --config experiments/heads.json --data-dir .cache/nltk_data --model-cache .cache/models --output runs/heads.json
python -m dependency_parsing run --config experiments/combine.json --data-dir .cache/nltk_data --model-cache .cache/models --output runs/combine.json
```

Only the first command permits a model download; later commands require the pinned revision already cached. Expect roughly 440 MB for float32 BERT weights plus tokenizer/configuration files. Everything runs on CPU. There is one forward pass per sentence, with all layer/head attentions extracted together. The 144-head sweep performs 144 tree decodes per sentence and is substantially more expensive than three mean-head decodes. The combiner materializes candidate-edge features for 300 training sentences and the test attention cache; allow multiple GB of RAM. No measured end-to-end attention runtime is claimed for this release.

The model is fixed in eval mode with eager attention. Overlength inputs or missing word alignment stop the run; no sentences are silently skipped. `--allow-download` is an explicit request to acquire the configured public model. No private credentials or corpus text are written into the result JSON.

The JSON output records the exact input configuration, SHA-256, ordered corpus fingerprint, Python/library versions, elapsed wall time, and scores. A test oracle is always labeled test-selected. Logistic-regression iteration counts are saved; convergence warnings must be reviewed before treating a run as complete. The public historical table remains separate from newly generated results.
