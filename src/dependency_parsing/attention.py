"""Frozen-attention experiments, following the coursework's aggregation contract.

The supplied extractor is not redistributed. This vectorized implementation
follows the authored bonus-heads code and makes alignment failures explicit.
"""
import numpy as np

MODEL = 'google-bert/bert-base-uncased'
REVISION = 'b84905490c2d4796d85a42590c88a95af94cd5e2'


def aggregate_heads(attentions, word_ids, n_words, cls_position=0):
    """[layers, heads, pieces, pieces] -> [layers, heads, words+ROOT, words+ROOT].

    Mean over BOTH source and target subpieces. ROOT row uses CLS attention;
    ROOT column and diagonal are zero. Rows are not renormalized after removal
    of special tokens/self-attention. Direction is source head -> dependent.
    """
    a = np.asarray(attentions)
    if (a.ndim != 4 or a.shape[-2:] != (len(word_ids), len(word_ids))
            or not np.isfinite(a).all()):
        raise ValueError('invalid attention tensor or token alignment')
    if n_words < 0 or not 0 <= cls_position < len(word_ids):
        raise ValueError('invalid word count or CLS position')
    s = np.zeros((n_words, len(word_ids)))
    for w in range(n_words):
        positions = [p for p, wid in enumerate(word_ids) if wid == w]
        if not positions:
            raise ValueError('every input word must have at least one subtoken')
        s[w, positions] = 1 / len(positions)
    if any(w is not None and (w < 0 or w >= n_words) for w in word_ids):
        raise ValueError('word id outside input sentence')
    mats = np.zeros((*a.shape[:2], n_words + 1, n_words + 1))
    mats[:, :, 1:, 1:] = np.einsum('wp,LHpq,vq->LHwv', s, a, s, optimize=True)
    mats[:, :, 0, 1:] = np.einsum('wq,LHq->LHw', s, a[:, :, cls_position, :])
    d = np.arange(n_words + 1)
    mats[:, :, d, d] = 0
    return mats


def load_model(cache_dir, model_name=MODEL, revision=REVISION, allow_download=False):
    from transformers import AutoTokenizer, AutoModel
    kwargs = dict(cache_dir=str(cache_dir), revision=revision,
                  local_files_only=not allow_download)
    tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs,
                                     output_attentions=True, attn_implementation='eager')
    model.eval()
    return tok, model


def all_head_matrices(words, tokenizer, model):
    import torch
    enc = tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=False)
    if enc['input_ids'].shape[-1] > model.config.max_position_embeddings:
        raise ValueError('sentence exceeds model limit; no silent truncation')
    word_ids = enc.word_ids()
    cls = (enc['input_ids'][0] == tokenizer.cls_token_id).nonzero().flatten().tolist()
    if len(cls) != 1:
        raise ValueError('expected one CLS token')
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
    if outputs.attentions is None or any(a is None for a in outputs.attentions):
        raise ValueError('model must expose attention tensors; use eager attention')
    a = torch.stack(outputs.attentions)[:, 0].cpu().numpy()
    return aggregate_heads(a, word_ids, len(words), cls[0])


def attn_to_arc_scores(matrix):
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.isfinite(m).all():
        raise ValueError('scores require a finite square matrix including ROOT')
    return {(u, v): -float(m[u, v]) for v in range(1, len(m))
            for u in range(len(m)) if u != v}


def edges(n):
    return [(u, v) for v in range(1, n + 1) for u in range(n + 1) if u != v]


def edge_features(mats):
    """Forward then backward features, in layer-major/head-major order."""
    f = mats.reshape(-1, mats.shape[-2], mats.shape[-1])
    return np.asarray([np.concatenate((f[:, u, v], f[:, v, u]))
                       for u, v in edges(f.shape[-1] - 1)], dtype=np.float32)


def head_grid(data, tok, model):
    from .decoder import cle_min
    from .perceptron import unlabeled_attachment_score as uas
    totals, means = None, None
    if not data:
        raise ValueError('empty head-selection/evaluation data')
    for tokens, gold in data:
        mats = all_head_matrices([t.word for t in tokens[1:]], tok, model)
        if totals is None:
            totals = np.zeros(mats.shape[:2]); means = np.zeros(mats.shape[0])
        for l in range(mats.shape[0]):
            for h in range(mats.shape[1]):
                totals[l, h] += uas(cle_min(attn_to_arc_scores(mats[l, h]), len(tokens)), gold)
            means[l] += uas(cle_min(attn_to_arc_scores(mats[l].mean(0)), len(tokens)), gold)
    return totals / len(data), means / len(data)
