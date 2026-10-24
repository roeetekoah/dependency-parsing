# Experiment ledger

## Shared protocol

The local NLTK dependency treebank contains 3,914 sentences. Ordered `.dp` files and sentence order match `dependency_treebank.parsed_sents()`; the first `int(3914 × 0.9)=3522` sentences train, and the last 392 test. This is an ordered, sentence-level split, not a random or document-disjoint split. No validation set is used. All token types, including punctuation, count. Gold POS tags are available to the classical parser; no tagger is trained.

The historical metric is **sentence-macro UAS**: average `correct_heads / sentence_tokens` over test sentences. Long and short sentences have equal weight. New classical/mean-attention records additionally label token-micro UAS and token-weighted diagnostics by gold edge distance, sentence length, and ROOT status. Relation labels are absent from the three-column source data, so no relation-level breakdown is claimed.

Each word has one parent, ROOT never has a parent, and multiple ROOT children are allowed. Arc scores are negated for minimum decoding. There is no projectivity constraint. The public release uses deterministic local ties, which may change a perceptron trajectory relative to the historical decoder.

## Configurations

| File | Training / selection | Evaluation |
|---|---|---|
| `classical.json` | All 3,522 train sentences, two epochs, rate 1, seed 0, word+POS pairs | Full test split |
| `robustness.json` | Same, seeds 0–4; only shuffle order varies | Full test; mean and sample SD |
| `attention.json` | Frozen BERT; no parser fitting, mean across heads, layers 0/5/11 | Full test, all three settings reported |
| `heads.json` | Best of all layer/head pairs selected on first 250 train sentences; flatten-order argmax ties | Full test; train-selected head reported separately from test-selected oracle |
| `combine.json` | First 300 train sentences; all candidate arcs labeled gold/not-gold; balanced logistic regression, C=1, max_iter=2000, lbfgs | Full test; logits decoded as one global tree; mean/max controls reported too |

The learned combiner uses 144 forward and 144 reverse features for BERT-base (12 layers × 12 heads). ROOT's reverse features are zero because the extraction creates no incoming ROOT column. Values are mean attention over both source and target subpieces, with [CLS]→word attention as a root proxy. [SEP] is discarded, self-attention is zeroed, and rows are not renormalized. The training-free readout uses attention source as candidate syntactic head, an assumption rather than a fact about BERT. Layer 0 is the first encoder layer, 11 the last. No transposition, summing over target pieces, tokenizer-word averaging change, or fine-tuning is hidden in the release.

New attention runs pin the model revision in their configuration and request eager attention. The historical model revision, runtime versions, hardware details, and full optimizer convergence record are unknown. A future pinned run is therefore a new, comparable experiment, not a recovered bit-for-bit historical environment.

## Historical evidence quality

`historical-results.csv` is the normalized numerical ledger. `results.txt` supports four corrected-decoder and four supplied-decoder values to four decimal places. Bonus experiments, ablations, baselines, and robustness are supported by the report only, generally at lower precision; no fabricated raw grid or per-seed precision is supplied. The reported mean/SD is retained as written; it must not be recomputed from rounded per-seed table values.

The head sweep's code selects on training data. Its per-layer best-head curve and test oracle are test-selected diagnostics, not independent selection estimates. The report says training and test selection both identify L3/H5; that is historical report evidence, not a recovered raw grid. The lowest/highest-layer observations do not establish why the result occurs. The learned combiner is supervised and uses additional backward features, so it is not an apples-to-apples estimate of the effect of combining heads alone.

The report's neighbour baselines need a further tree-validity qualification: attaching to an adjacent token can be defined as a chain, but the historical random-head procedure and precise boundary handling are not recorded in runnable retained baseline scripts. These values are contextual report evidence, not release benchmarks. Ablations and averaged-vs-final comparisons likewise have no retained dedicated runner here. The focused release does not manufacture configurations for those rows.

## Interpretation limits

The five seeds assess sensitivity to sentence order only. They do not assess corpus sampling uncertainty or prove two epochs are sufficient for convergence. The last-10% split has no validation stage and was used to compare settings; selecting a winning layer from it is exploratory. No statistical significance claim, optimized parser claim, universal middle-layer claim, causal attention explanation, or state-of-the-art claim follows from these results.

## New classical verification

`release-classical-run.json` records a new five-seed release run: 0.410848 mean macro UAS and 0.002548 sample SD, 3,522 training and 392 test sentences (9,172 test tokens), approximately 129 seconds. Seed-0 macro/micro values are 0.412655/0.370039. A separate private check reproduced legacy seed 0 at 0.405036 with exactly identical loaded examples. Decoder tie policy changes can alter zero-initialized predictions and subsequent perceptron updates; the new value is not evidence of an accuracy gain over the old algorithm. The later registered NLTK path-root and exact-integer normalization changes do not alter this float-scored training run.

The new attention implementation passed a no-download integration test with a tiny randomly initialized BERT for all three experiment routes. That validates mechanics and library compatibility, not the pretrained-model numbers. Full pretrained-BERT reruns were not performed; the model was not present in the inspected cache.
