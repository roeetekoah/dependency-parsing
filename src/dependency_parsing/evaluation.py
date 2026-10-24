"""Sentence-macro UAS with separately labeled token and error diagnostics."""
from collections import defaultdict


def attachment_report(data, predictions):
    if not data or len(data) != len(predictions):
        raise ValueError('nonempty, aligned gold/prediction sequences required')
    macro, correct, count = 0., 0, 0
    bins = defaultdict(lambda: [0, 0])
    for (tokens, gold), pred in zip(data, predictions):
        if not gold:
            raise ValueError('gold sentence must not be empty')
        hit = sum(pred.get(v) == h for v, h in gold.items())
        macro += hit / len(gold); correct += hit; count += len(gold)
        for v, h in gold.items():
            d = abs(v - h)
            labels = ['root' if h == 0 else 'non_root',
                      'length_1_10' if len(gold) <= 10 else
                      'length_11_20' if len(gold) <= 20 else 'length_21_plus']
            if h != 0:
                labels.append('distance_1' if d == 1 else 'distance_2_5' if d <= 5 else 'distance_6_plus')
            for label in labels:
                bins[label][0] += int(pred.get(v) == h); bins[label][1] += 1
    return {'macro_uas': macro / len(data), 'micro_uas': correct / count,
            'sentences': len(data), 'tokens': count,
            'diagnostics': {k: {'correct': c, 'tokens': n, 'micro_uas': c / n}
                            for k, (c, n) in sorted(bins.items())}}
