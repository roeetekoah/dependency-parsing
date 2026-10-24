"""Minimum rooted arborescence, adapted from the coursework correction.

The contraction/expansion design follows chu_liu_edmonds_fixed.py; later
release changes add validation, deterministic ties, linear incoming-edge scans,
and an explicit contraction stack. See docs/decoder.md and docs/provenance.md.
"""
from math import isfinite
from numbers import Integral, Real


class NoArborescenceError(ValueError):
    """The finite allowed edges cannot connect every node to ROOT."""


def _cycle(parent, nodes):
    done = {0}
    for start in nodes:
        path, positions, v = [], {}, start
        while v not in done and v not in positions:
            positions[v] = len(path)
            path.append(v)
            v = parent[v]
        if v in positions:
            return path[positions[v]:]
        done.update(path)
    return None


def cle_min(scores_dict, n):
    """Return {child: parent} minimizing sum(scores[parent, child]).

    Nodes are 0..n-1, ROOT=0. Missing and +inf arcs are forbidden. Self-loops
    and arcs into ROOT are ignored after validation. NaN, -inf, non-real costs,
    invalid endpoints, and n<1 raise ValueError. Infeasibility raises
    NoArborescenceError. Multiple children of ROOT are permitted. Equal-cost
    choices are deterministic under insertion-order changes, not globally
    lexicographically minimal. Input is never modified.
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    scores = {}
    for edge, weight in scores_dict.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("edge keys must be (parent, child) pairs")
        u, v = edge
        if any(isinstance(x, bool) or not isinstance(x, Integral)
               or x < 0 or x >= n for x in edge):
            raise ValueError("edge endpoint outside 0..n-1")
        if not isinstance(weight, Real) or weight != weight or weight == -float('inf'):
            raise ValueError("cost must be real, without NaN or negative infinity")
        if u != v and v != 0 and weight != float('inf'):
            # Preserve exact integer arithmetic, including NumPy integer inputs.
            scores[(int(u), int(v))] = int(weight) if isinstance(weight, Integral) else weight
    scores = dict(sorted(scores.items()))
    nodes = tuple(range(n))
    stack = []
    while True:
        incoming = {}
        for (u, v), w in scores.items():
            if v not in incoming or (w, u) < (incoming[v][1], incoming[v][0]):
                incoming[v] = (u, w)
        if any(v not in incoming for v in nodes if v != 0):
            raise NoArborescenceError("no arborescence rooted at node 0")
        parent = {v: incoming[v][0] for v in nodes if v != 0}
        cycle = _cycle(parent, nodes)
        if cycle is None:
            break
        cyc = set(cycle)
        vc = max(nodes) + 1
        new_scores, in_to, out_from = {}, {}, {}
        for (u, v), w in scores.items():
            u_in, v_in = u in cyc, v in cyc
            if u_in and v_in:
                continue
            key = (vc if u_in else u, vc if v_in else v)
            adjusted = w - incoming[v][1] if v_in else w
            if not isinstance(adjusted, Integral) and not isfinite(adjusted):
                raise ValueError("cost arithmetic overflow during contraction")
            # The original edge is the tie-breaker for collapsed parallel arcs.
            candidate = (adjusted, u, v)
            if key not in new_scores or candidate < new_scores[key]:
                new_scores[key] = candidate
                if v_in:
                    in_to[key] = v
                if u_in:
                    out_from[key] = u
        stack.append((vc, cyc, parent, in_to, out_from))
        scores = {key: value[0] for key, value in new_scores.items()}
        nodes = tuple(v for v in nodes if v not in cyc) + (vc,)

    for vc, cyc, old_parent, in_to, out_from in reversed(stack):
        external = parent[vc]
        target = in_to[(external, vc)]
        expanded = {v: out_from[(vc, v)] if u == vc else u
                    for v, u in parent.items() if v != vc}
        expanded.update({v: old_parent[v] for v in cyc if v != target})
        expanded[target] = external
        parent = expanded
    return parent
