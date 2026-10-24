"""
perceptron.py
=============
Classical arc-factored parsing, adapted from joint coursework: an MST dependency parser trained with the averaged
perceptron (Lecture 9 - Dependency Parsing).

Pipeline
--------
  load_data()                  -> train/test split of (tokens, gold_heads)
  feature_function()           -> Boolean edge features
  train_averaged_perceptron()  -> learn the weights
  predict()                    -> run Chu-Liu/Edmonds on the learned scores
  unlabeled_attachment_score() -> UAS for one sentence
  evaluate()                  -> mean sentence UAS (CLI owns experiment runs)

Data structures
-----------------------------------------------------
A sentence is a list[Token] where index 0 is an artificial ROOT
(word='ROOT', tag='ROOT') and indices 1..n are the words.
gold_heads is a dict {child_index: head_index}; head 0 means "attaches to ROOT".
A predicted parse has the same {child: head} shape (this is what cle_min returns).
"""
from collections import defaultdict
from collections.abc import Callable, Mapping
from enum import Enum, auto
from typing import NamedTuple

import random
import time
import math

from .decoder import cle_min
from .data import load_data
ROOT = 0


class Token(NamedTuple):
    word: str
    tag: str


class Feat(Enum):
    """Which template a feature key belongs to (its namespace tag)."""
    WORD = auto()   # word bigram: (Feat.WORD, head_word, dep_word)
    POS = auto()    # POS  bigram: (Feat.POS,  head_tag,  dep_tag)


# Type aliases used throughout.
Sentence = list[Token]            # token 0 is ROOT, tokens 1..n are the words
Heads = dict[int, int]            # child index -> head index (head 0 = ROOT)
Example = tuple[Sentence, Heads]  # one (sentence, gold parse) pair
FeatureKey = tuple[Feat, str, str]
Weights = Mapping[FeatureKey, float]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Feature function
# --------------------------------------------------------------------------- #
def feature_function(u: int, v: int, tokens: Sentence) -> list[FeatureKey]:
    """Return the list of *active* (Boolean) features for the edge u -> v.

    Parameters
    ----------
    u : int            index of the candidate head       (0 = ROOT)
    v : int            index of the candidate dependent
    tokens : list[Token]   the sentence (token 0 is ROOT)

    Returns
    -------
    list[tuple]         the two feature keys that fire for this edge.

    The full indicator vector f in {0,1}^d is almost all zeros: for any edge
    exactly two coordinates are 1 -- the word pair (word(u), word(v)) and the
    tag pair (tag(u), tag(v)) -- so we never build it. We return just those two
    active keys and score_edge sums their weights. Each key is a tuple tagged by
    its template (Feat.WORD / Feat.POS) so the two families never share a weight
    -- no magic strings, no delimiter to escape.

      * word bigram : (Feat.WORD, head word form, dependent word form)
      * POS  bigram : (Feat.POS,  head POS tag,   dependent POS tag)
    """
    return [(Feat.WORD, tokens[u].word, tokens[v].word),
            (Feat.POS,  tokens[u].tag,  tokens[v].tag)]


# --------------------------------------------------------------------------- #
# Scoring + inference
# --------------------------------------------------------------------------- #
def score_edge(u: int, v: int, tokens: Sentence, weights: Weights) -> float:
    """Score of edge u -> v = sum of weights of its active features.

    Uses .get(f, 0.0) so the plain dict returned by training works at eval time:
    a feature unseen in training is simply absent and is 0 by definition.
    """
    return sum(weights.get(f, 0.0) for f in feature_function(u, v, tokens))


def predict(tokens: Sentence, weights: Weights) -> Heads:
    """Highest-scoring tree as {child: head}, via Chu-Liu/Edmonds.

    cle_min finds the *minimum* arborescence, so we negate the scores to get
    the *maximum* spanning tree.  ROOT (index 0) is never a dependent.
    """
    n = len(tokens) - 1                       # number of words (excludes ROOT)
    scores = {}
    for v in range(1, n + 1):
        for u in range(0, n + 1):
            if u == v:
                continue

            scores[(u, v)] = -score_edge(u, v, tokens, weights)

    return cle_min(scores, n + 1)


# --------------------------------------------------------------------------- #
# Progress logging (kept out of the core training logic below)
# --------------------------------------------------------------------------- #
ProgressFn = Callable[[int, int, int, int], None]   # (done, total, mistakes, |w|)


def _progress_reporter(every: int = 500) -> ProgressFn:
    """Return a callback that prints a live one-line progress bar."""
    def report(done: int, total: int, mistakes: int, n_features: int) -> None:
        if done % every == 0 or done == total:
            end = "\n" if done == total else ""
            print(f"\r    {done}/{total} sentences  mistakes={mistakes}  "
                  f"|w|={n_features}", end=end, flush=True)
    return report


def _log_epoch(epoch: int, n_epochs: int, mistakes: int, n_sents: int,
               n_features: int, secs: float) -> None:
    """Print a one-line per-epoch summary."""
    print(f"  epoch {epoch}/{n_epochs}: {mistakes}/{n_sents} mis-parsed, "
          f"|w|={n_features}, {secs:.1f}s")


# --------------------------------------------------------------------------- #
# Training  (pure logic: progress is emitted through an optional callback)
# --------------------------------------------------------------------------- #
def _apply(weights: dict[FeatureKey, float], totals: dict[FeatureKey, float],
           heads: Heads, tokens: Sentence, delta: float, t: int) -> None:
    """Nudge every active feature of every edge in `heads` by `delta`.

    Updates the live weights (`weights[f] += delta`) and, in the same pass, the
    lazy-averaging accumulator (`totals[f] += delta * t`) at global step `t`.
    Tracking delta*t lets train() reconstruct the average of the weight vector
    over all steps in O(1) per update -- avg = ((T+1)*weights[f] - totals[f]) / T
    -- instead of summing the whole vector after every sentence.
    """
    for child, head in heads.items():        # an edge is head -> child
        for f in feature_function(head, child, tokens):
            weights[f] += delta
            totals[f] += delta * t


def _perceptron_pass(train_data: list[Example], w: dict[FeatureKey, float],
                     totals: dict[FeatureKey, float], lr: float, t0: int,
                     report: ProgressFn | None = None, rng=None) -> int:
    """One epoch starting at global step `t0`. Returns #mis-parsed.

    No printing here: if `report` is given it is called once per sentence as
    report(done, total, mistakes, n_features); the reporter decides what to show.
    """
    shuffled = list(train_data)
    (rng or random).shuffle(shuffled)
    n = len(shuffled)
    mistakes = 0

    for i, (tokens, gold_heads) in enumerate(shuffled, start=1):
        t = t0 + i
        pred_heads = predict(tokens, w)
        if pred_heads != gold_heads:
            mistakes += 1
        _apply(w, totals, gold_heads, tokens, +lr, t)
        _apply(w, totals, pred_heads, tokens, -lr, t)
        if report is not None:
            report(i, n, mistakes, len(w))

    return mistakes


def train_averaged_perceptron(train_data: list[Example], n_iterations: int = 2,
                              lr: float = 1.0, seed: int = 0,
                              verbose: bool = True) -> dict[FeatureKey, float]:
    """Learn and return the averaged-perceptron weight vector.

    The averaged weights are the mean of the weight vector over all
    T = n_iterations * len(train_data) update steps. Pass verbose=False to
    silence all progress output.

    Notes
    -----
    * gold_heads is {child: head}; an edge is head -> child.
    * Averaging is done lazily: instead of summing the whole weight vector after
      every sentence (O(|w|) per step), we accumulate delta*t per touched
      feature and reconstruct avg[f] = ((T+1)*w[f] - totals[f]) / T at the end.
      Identical result, O(1) per update.
    """
    if not train_data or isinstance(n_iterations, bool) or not isinstance(n_iterations, int) or n_iterations < 1:
        raise ValueError("training data and iteration count must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    rng = random.Random(seed)
    w, totals = defaultdict(float), defaultdict(float)
    report = _progress_reporter() if verbose else None
    n_sents = len(train_data)

    for it in range(1, n_iterations + 1):
        tic = time.perf_counter()
        mistakes = _perceptron_pass(train_data, w, totals, lr,
                                    (it - 1) * n_sents, report, rng)
        if verbose:
            _log_epoch(it, n_iterations, mistakes, n_sents, len(w),
                       time.perf_counter() - tic)

    T = n_iterations * n_sents
    return {f: ((T + 1) * w[f] - totals[f]) / T for f in w}

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def unlabeled_attachment_score(pred_heads: Heads, gold_heads: Heads) -> float:
    """Fraction of words whose predicted head matches the gold head (UAS)."""
    if not gold_heads:
        raise ValueError("UAS requires at least one gold token")
    correct = sum(1 for v, h in gold_heads.items() if pred_heads.get(v) == h)
    return correct / len(gold_heads)


def evaluate(data: list[Example], weights: Weights) -> float:
    """Mean UAS over a dataset."""
    if not data:
        raise ValueError("evaluation data must not be empty")
    return sum(
        unlabeled_attachment_score(predict(tokens, weights), gold_heads)
        for tokens, gold_heads in data
    ) / len(data)


