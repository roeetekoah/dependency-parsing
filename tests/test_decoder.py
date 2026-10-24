"""Independent exhaustive oracle: enumerate parents, then check root reachability."""
from itertools import product
import math
import random

import pytest

from dependency_parsing.decoder import cle_min, NoArborescenceError


def valid(tree, scores, n):
    if set(tree) != set(range(1, n)):
        return False
    for child in range(1, n):
        seen, v = set(), child
        while v != 0:
            if v in seen or v not in tree or (tree[v], v) not in scores:
                return False
            seen.add(v); v = tree[v]
    return True


def oracle(scores, n):
    choices = [[u for u in range(n) if u != v and (u, v) in scores
                and math.isfinite(scores[u, v])] for v in range(1, n)]
    best = None
    for parents in product(*choices):
        tree = dict(zip(range(1, n), parents))
        if valid(tree, scores, n):
            cost = sum(scores[u, v] for v, u in tree.items())
            best = cost if best is None else min(best, cost)
    return best


def check(scores, n):
    optimum = oracle(scores, n)
    if optimum is None:
        with pytest.raises(NoArborescenceError):
            cle_min(scores, n)
    else:
        tree = cle_min(scores, n)
        assert valid(tree, scores, n), (scores, tree)
        assert sum(scores[u, v] for v, u in tree.items()) == pytest.approx(optimum), (scores, tree)


def test_known_outgoing_cycle_regression():
    scores = {(0, 1): 10, (0, 2): 10, (0, 3): 10,
              (2, 1): 1, (1, 2): 1, (2, 3): 1, (1, 3): 5}
    tree = cle_min(scores, 4)
    assert tree[3] == 2
    assert sum(scores[u, v] for v, u in tree.items()) == 12


def test_exhaustive_four_node_sparse_weighted_graphs():
    # 3**9 = 19,683 graphs: absent, -1 or +2 for every legal arc.
    edges = [(u, v) for v in range(1, 4) for u in range(4) if u != v]
    for values in product([None, -1, 2], repeat=len(edges)):
        check({e: w for e, w in zip(edges, values) if w is not None}, 4)


def test_random_dense_sparse_and_float_graphs():
    rng = random.Random(20261011)
    for i in range(600):
        n = rng.randint(2, 6)
        scores = {(u, v): rng.randint(-9, 9) / (4 if i % 3 == 0 else 1)
                  for v in range(1, n) for u in range(n)
                  if u != v and (i % 2 == 0 or rng.random() < .55)}
        check(scores, n)


def test_nested_contraction():
    # 1<->2 contracts first; its supernode then cycles with node 3.
    s = {(0, 1): 8, (0, 2): 9, (0, 3): 9,
         (1, 2): 0, (2, 1): 0, (2, 3): 1, (3, 1): 1}
    check(s, 4)


def test_ties_insertion_order_and_no_mutation():
    s = {(u, v): 0 for v in range(1, 5) for u in range(5) if u != v}
    before = s.copy()
    assert cle_min(s, 5) == cle_min(dict(reversed(list(s.items()))), 5)
    assert s == before
    assert cle_min(s, 5) == {1: 0, 2: 0, 3: 0, 4: 0}


def test_incoming_shift_invariance_and_maximum_sign():
    s = {(0, 1): 6, (0, 2): 8, (1, 2): 2, (2, 1): 1}
    shifted = {(u, v): w + v * 7 for (u, v), w in s.items()}
    assert cle_min(s, 3) == cle_min(shifted, 3)
    negative = {e: -w for e, w in s.items()}
    assert sum(s[u, v] for v, u in cle_min(negative, 3).items()) == 14


def test_large_integer_contraction_preserves_exact_costs():
    base = 10 ** 400
    s = {(0, 1): base + 10, (0, 2): base + 10,
         (1, 2): base, (2, 1): base}
    tree = cle_min(s, 3)
    assert sum(s[u, v] for v, u in tree.items()) == 2 * base + 10


def test_singleton_ignored_edges_infinite_and_disconnected():
    assert cle_min({}, 1) == {}
    assert cle_min({(1, 0): -99, (1, 1): -99, (0, 1): 2}, 2) == {1: 0}
    assert cle_min({(0, 1): math.inf, (0, 2): 1, (2, 1): 3}, 3) == {1: 2, 2: 0}
    with pytest.raises(NoArborescenceError):
        cle_min({(1, 2): 0, (2, 1): 0}, 3)
    with pytest.raises(NoArborescenceError):
        cle_min({(0, 1): math.inf}, 2)


@pytest.mark.parametrize('scores,n', [({}, 0), ({}, True), ({}, 1.5),
    ({(-1, 1): 0}, 2), ({(0, 2): 0}, 2), ({(0, True): 0}, 2),
    ({(0, 1): float('nan')}, 2), ({(0, 1): -math.inf}, 2),
    ({(0, 1): '1'}, 2), ({'01': 1}, 2)])
def test_invalid_inputs(scores, n):
    with pytest.raises(ValueError):
        cle_min(scores, n)
