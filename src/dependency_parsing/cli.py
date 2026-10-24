"""Download-free demo and explicit, configuration-driven experiment entrypoint."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

from .data import load_data, synthetic_data, corpus_fingerprint
from .perceptron import train_averaged_perceptron, predict
from .evaluation import attachment_report


def classical(train, test, cfg):
    records = []
    for seed in cfg['seeds']:
        weights = train_averaged_perceptron(train, cfg['epochs'], cfg['learning_rate'], seed, False)
        records.append({'seed': seed, **attachment_report(test, [predict(t, weights) for t, _ in test])})
    values = [r['macro_uas'] for r in records]
    return {'runs': records, 'mean_macro_uas': statistics.fmean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None}


def attention_experiment(train, test, cfg, args):
    import numpy as np
    from .attention import load_model, all_head_matrices, attn_to_arc_scores, head_grid, edge_features, edges
    from .decoder import cle_min
    from .perceptron import unlabeled_attachment_score as uas
    tok, model = load_model(args.model_cache, cfg['model'], cfg['revision'], args.allow_download)
    if cfg['kind'] == 'attention':
        preds = {l: [] for l in cfg['layers']}
        for tokens, _ in test:
            mats = all_head_matrices([t.word for t in tokens[1:]], tok, model)
            for l in cfg['layers']:
                preds[l].append(cle_min(attn_to_arc_scores(mats[l].mean(0)), len(tokens)))
        return {str(l): attachment_report(test, p) for l, p in preds.items()}
    if cfg['kind'] == 'head-sweep':
        subset = train[:cfg['train_subset']]
        grid_train, _ = head_grid(subset, tok, model)
        selected = tuple(int(x) for x in np.unravel_index(np.argmax(grid_train), grid_train.shape))
        grid_test, means = head_grid(test, tok, model)
        oracle = tuple(int(x) for x in np.unravel_index(np.argmax(grid_test), grid_test.shape))
        return {'train_selected_layer_head': selected, 'test_macro_uas': float(grid_test[selected]),
                'train_selection_sentences': len(subset), 'test_grid': grid_test.tolist(),
                'mean_head_test_macro_uas': means.tolist(),
                'test_oracle_layer_head': oracle, 'test_oracle_macro_uas': float(grid_test[oracle]),
                'oracle_status': 'test-selected diagnostic, not a generalization estimate'}
    from sklearn.linear_model import LogisticRegression
    def extract(data):
        return [(all_head_matrices([t.word for t in tokens[1:]], tok, model), gold)
                for tokens, gold in data]
    tr = extract(train[:cfg['train_subset']])
    x = np.concatenate([edge_features(f) for f, _ in tr])
    y = np.concatenate([[int(gold[v] == u) for u, v in edges(f.shape[-1] - 1)] for f, gold in tr])
    clf = LogisticRegression(**cfg['logistic_regression']).fit(x, y)
    totals = {'max_all_heads': 0., 'mean_all_heads': 0., 'learned_combiner': 0.}
    for f, gold in extract(test):
        flat = f.reshape(-1, *f.shape[-2:])
        for name, m in [('max_all_heads', flat.max(0)), ('mean_all_heads', flat.mean(0))]:
            totals[name] += uas(cle_min(attn_to_arc_scores(m), len(m)), gold)
        logits = clf.decision_function(edge_features(f))
        scores = {e: -float(s) for e, s in zip(edges(f.shape[-1] - 1), logits)}
        totals['learned_combiner'] += uas(cle_min(scores, f.shape[-1]), gold)
    return {'macro_uas': {k: v / len(test) for k, v in totals.items()},
            'train_subset_sentences': len(tr), 'fit_edges': len(y),
            'positive_edges': int(y.sum()), 'features': x.shape[1],
            'optimizer_iterations': clf.n_iter_.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo', 'run'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-cache', type=Path, default=Path('.cache/models'))
    parser.add_argument('--allow-download', action='store_true', help='allow public model download')
    args = parser.parse_args()
    start = time.perf_counter()
    if args.command == 'demo':
        data = synthetic_data()
        result = {'status': 'synthetic demonstration; training and evaluation on same tiny fixture',
                  **classical(data, data, {'epochs': 6, 'learning_rate': 1., 'seeds': [0]})}
    else:
        if not args.config or not args.data_dir:
            parser.error('run requires --config and --data-dir')
        raw = args.config.read_bytes(); cfg = json.loads(raw)
        if cfg.get('kind') not in ['classical', 'attention', 'head-sweep', 'combine']:
            parser.error('unsupported experiment kind')
        contract = {'metric': 'sentence_macro_uas', 'punctuation': 'included',
                    'decoder': 'release_deterministic', 'root': 'node_0_multiple_children_allowed'}
        if any(cfg.get(k) != v for k, v in contract.items()):
            parser.error('configuration does not match the implemented metric/decoder contract')
        if cfg['kind'] != 'classical':
            attention_contract = {'attention_direction': 'head_to_dependent',
                'subword_pooling': 'mean_both_axes', 'root_proxy': 'CLS_row',
                'device': 'cpu', 'attention_backend': 'eager'}
            if any(cfg.get(k) != v for k, v in attention_contract.items()):
                parser.error('configuration does not match the implemented attention contract')
        train, test = load_data(cfg['test_fraction'], args.data_dir)
        if 'train_subset' in cfg and not 1 <= cfg['train_subset'] <= len(train):
            parser.error('train_subset must be within training split')
        scores = classical(train, test, cfg) if cfg['kind'] == 'classical' else attention_experiment(train, test, cfg, args)
        result = {'status': 'new release run; not a recovered historical environment',
                  'configuration': cfg, 'config_sha256': hashlib.sha256(raw).hexdigest(),
                  'corpus_sha256': corpus_fingerprint(args.data_dir),
                  'train_sentences': len(train), 'test_sentences': len(test), 'results': scores}
    result['elapsed_seconds'] = time.perf_counter() - start
    result['python'] = platform.python_version()
    result['dependencies'] = {}
    for name in ['nltk', 'numpy', 'torch', 'transformers', 'scikit-learn']:
        try:
            result['dependencies'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
