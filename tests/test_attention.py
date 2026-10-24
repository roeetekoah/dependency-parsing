import numpy as np
import pytest
from types import SimpleNamespace

from dependency_parsing.attention import aggregate_heads, attn_to_arc_scores, edge_features
from dependency_parsing.decoder import cle_min
from dependency_parsing.cli import attention_experiment
from dependency_parsing import attention
from dependency_parsing.data import synthetic_data


def test_subword_mean_cls_root_and_direction_against_explicit_loop():
    rng = np.random.default_rng(42)
    a = rng.random((2, 3, 6, 6))
    ids = [None, 0, 1, 1, 2, None]
    actual = aggregate_heads(a, ids, 3)
    expected = np.zeros((2, 3, 4, 4))
    groups = [[1], [2, 3], [4]]
    for l in range(2):
        for h in range(3):
            for v, cols in enumerate(groups, 1):
                expected[l, h, 0, v] = a[l, h, 0, cols].mean()
                for u, rows in enumerate(groups, 1):
                    if u != v:
                        expected[l, h, u, v] = np.mean([a[l, h, r, c] for r in rows for c in cols])
    np.testing.assert_allclose(actual, expected)
    # Mean-head pooling commutes with the word aggregation.
    np.testing.assert_allclose(actual.mean(1), aggregate_heads(a.mean(1, keepdims=True), ids, 3)[:, 0])
    assert np.all(actual[:, :, :, 0] == 0)


def test_sign_and_root_are_structural():
    m = np.array([[0., .1, .9], [.9, 0., .2], [.8, .7, 0.]])
    s = attn_to_arc_scores(m)
    assert cle_min(s, 3) == {1: 2, 2: 0}
    assert not any(v == 0 or u == v for u, v in s)


def test_forward_backward_feature_order_and_root_zero_block():
    f = np.arange(2 * 2 * 3 * 3).reshape(2, 2, 3, 3).astype(float)
    f[:, :, :, 0] = 0
    features = edge_features(f)
    flat = f.reshape(4, 3, 3)
    np.testing.assert_array_equal(features[0], np.concatenate([flat[:, 0, 1], flat[:, 1, 0]]))
    assert np.all(features[0, 4:] == 0)


def test_alignment_and_matrix_validation():
    with pytest.raises(ValueError):
        aggregate_heads(np.ones((1, 1, 2, 2)), [None, 0], 2)
    with pytest.raises(ValueError):
        attn_to_arc_scores(np.ones((2, 3)))
    with pytest.raises(ValueError):
        attn_to_arc_scores(np.array([[0., np.nan], [0., 0.]]))


def test_head_is_selected_on_training_even_when_test_oracle_differs(monkeypatch):
    train, test = synthetic_data()[:2], synthetic_data()[2:]
    monkeypatch.setattr(attention, 'load_model', lambda *a, **kw: (None, None))
    calls = []
    def grid(data, *args):
        calls.append(data)
        if len(data) == 2:
            return np.array([[0.1, 0.8], [0.2, 0.3]]), np.zeros(2)
        return np.array([[0.3, 0.12], [0.9, 0.2]]), np.zeros(2)
    monkeypatch.setattr(attention, 'head_grid', grid)
    result = attention_experiment(train, test,
        dict(kind='head-sweep', train_subset=2, model='unused', revision='unused'),
        SimpleNamespace(model_cache='unused', allow_download=False))
    assert calls == [train, test]
    assert result['train_selected_layer_head'] == (0, 1)
    assert result['test_macro_uas'] == .12
    assert result['test_oracle_layer_head'] == (1, 0)
    assert result['test_oracle_macro_uas'] == .9
