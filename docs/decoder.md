# Decoder contract and correctness evidence

`cle_min(costs, n)` finds a minimum-cost directed arborescence rooted at node 0. An arc key `(u, v)` means head/parent `u` → child `v`; the returned mapping uses the reverse storage convention `{child: parent}`. For maximum-score parsing, negate each score before calling it.

Every non-root node has one parent, and every parent chain reaches 0. ROOT may have multiple children. No projectivity constraint or single-root-child constraint is imposed.

| Input situation | Release behavior |
|---|---|
| `n=1`, no arcs | Empty tree, valid ROOT-only graph |
| `n<1`, non-integer node count, bad endpoints | `ValueError` |
| Missing arc or `+inf` cost | Forbidden arc |
| NaN, `-inf`, non-real cost | `ValueError` |
| Self-loop or incoming arc to ROOT | Ignored after validation |
| No rooted spanning arborescence | `NoArborescenceError`, never a partial tree |
| Equal-cost alternatives | Deterministic local tie choices; no global lexicographic guarantee |
| Cost subtraction overflows | `ValueError`; rescale finite floating-point inputs |

Input mappings are not modified. Integer costs retain exact integer arithmetic; ordinary finite floating-point costs use no epsilon tie threshold; very close costs can still be affected by roundoff. Modest finite floating-point costs are the intended floating-point regime.

## Why contraction works

Choose the cheapest incoming arc for each non-root node. If these arcs contain no directed cycle, they form a rooted arborescence: a finite parent chain either reaches the only node without an incoming arc (ROOT) or cycles, and cycles were excluded.

For a selected cycle C, contract it to a supernode. If an outside arc `u→v` enters C, subtract the cost of the selected cycle arc entering v. The remaining selected cycle costs contribute the same additive constant to every expanded solution. The adjusted weight therefore captures the cost of replacing that one cycle arc with `u→v`. A cycle-leaving arc keeps its original cost. Parallel contracted arcs retain the cheapest adjusted representative and its original endpoint. Solve the contracted graph, then expand: replace exactly the selected cycle arc entering the chosen entry target, restore the other cycle arcs, and restore the true source of every chosen outgoing arc.

This argument explains the implementation's invariant; the tests provide independent finite evidence, not a formal machine-checked proof.

## The supplied-code defect

For arcs `{0→1:10, 0→2:10, 0→3:10, 2→1:1, 1→2:1, 2→3:1, 1→3:5}`, nodes 1 and 2 initially cycle. The best outgoing arc to 3 is **2→3**, cost 1. The supplied decoder forgot that source and reconstructed **1→3**, cost 5. Its total becomes 16; the optimum is 12. The original corrected edition retained `out_from` as well as `in_to`. The release preserves this correction and adds interface validation and deterministic choices.

## Complexity of this implementation

Let V count ROOT and words, and E count finite allowed arcs. Canonical input sorting costs O(E log E). Each contraction scans the current edges once, finds incoming minima and a cycle in O(E+V), and removes at least one node. The explicit stack has at most V−1 contraction levels. Worst-case time is **O(E log E + V(E+V))**; for a feasible graph E≥V−1 this is O(E log E + VE), or **O(V³)** for dense parsing graphs. Saved contraction maps and parent maps use **O(V(E+V))** worst-case space, also cubic in dense graphs. This is a transparent implementation, not the optimized O(V²) dense algorithm discussed in parsing literature.

The legacy implementation scanned all edges separately for every destination and repeatedly walked parent chains; its Python implementation should not be described as the optimized textbook bound. Exhaustive testing is exponential and used only on tiny graphs.

## Tests

`tests/test_decoder.py` enumerates all 3⁹=19,683 four-node graphs whose legal arcs are absent, −1, or +2. An independent oracle enumerates one available parent per non-root node, checks every path to ROOT, and minimizes total original edge cost. Another 600 seeded sparse/dense graphs up to six nodes include negatives, ties, and exactly representable fractional costs. Named tests cover the known bug, nested contractions, graph infeasibility, ROOT-only input, ignored incoming/self arcs, forbidden infinities, insertion-order invariance, per-child cost shifts, and maximum-score sign reversal.

Reference: McDonald et al., [Non-Projective Dependency Parsing using Spanning Tree Algorithms](https://aclanthology.org/H05-1066/) (2005). The algorithmic framing is established prior work; the coursework contribution is its implementation, correction, and experiments.

The private edition audit also ran both legacy implementations on the 16,848 feasible graphs in the same exhaustive family: the coursework-corrected decoder produced no invalid/suboptimal trees; the supplied decoder produced 189 invalid trees and 125 valid but suboptimal trees. These counts describe this specific sparse graph family, not a general failure rate. The release does not ship the supplied code.
