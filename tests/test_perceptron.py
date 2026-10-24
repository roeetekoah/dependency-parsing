from collections import defaultdict
import random

import pytest

from dependency_parsing.data import synthetic_data, load_data
from dependency_parsing.perceptron import (train_averaged_perceptron, predict,
    feature_function, Token, unlabeled_attachment_score)
from dependency_parsing.evaluation import attachment_report


def test_lazy_average_matches_explicit_post_update_mean():
    data = synthetic_data(); epochs = 3; seed = 17
    rng, w, total = random.Random(seed), defaultdict(float), defaultdict(float)
    for _ in range(epochs):
        shuffled = list(data); rng.shuffle(shuffled)
        for tokens, gold in shuffled:
            pred = predict(tokens, w)
            for tree, delta in [(gold, 1), (pred, -1)]:
                for v, u in tree.items():
                    for f in feature_function(u, v, tokens):
                        w[f] += delta
            for f, value in w.items():
                total[f] += value
    expected = {f: value / (epochs * len(data)) for f, value in total.items()}
    actual = train_averaged_perceptron(data, epochs, seed=seed, verbose=False)
    assert actual == pytest.approx(expected)


def test_rng_is_local_and_training_is_repeatable():
    random.seed(44); state = random.getstate()
    a = train_averaged_perceptron(synthetic_data(), seed=2, verbose=False)
    assert random.getstate() == state
    b = train_averaged_perceptron(synthetic_data(), seed=2, verbose=False)
    assert a == b


def test_feature_namespaces_unseen_and_root_only():
    tokens = [Token('ROOT', 'ROOT'), Token('A', 'A')]
    assert len(set(feature_function(0, 1, tokens))) == 2
    assert predict(tokens, {}) == {1: 0}
    assert predict(tokens[:1], {}) == {}


def test_macro_micro_and_error_buckets():
    # Two short sentences: one perfect singleton and one 1/3 correct.
    data = [(None, {1: 0}), (None, {1: 0, 2: 1, 3: 1})]
    preds = [{1: 0}, {1: 0, 2: 0, 3: 0}]
    report = attachment_report(data, preds)
    assert report['macro_uas'] == pytest.approx(2 / 3)
    assert report['micro_uas'] == .5
    assert report['diagnostics']['root']['correct'] == 2
    assert report['diagnostics']['non_root']['correct'] == 0


def test_empty_data_rejected():
    with pytest.raises(ValueError):
        train_averaged_perceptron([], verbose=False)
    with pytest.raises(ValueError):
        unlabeled_attachment_score({}, {})
    with pytest.raises(ValueError):
        load_data(data_dir=None)
