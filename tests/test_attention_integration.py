"""Optional CPU integration with randomly initialized tiny BERT, no download."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
pytest.importorskip('sklearn')

from dependency_parsing import attention
from dependency_parsing.cli import attention_experiment
from dependency_parsing.data import synthetic_data


class Encoding(dict):
    def word_ids(self):
        return self.ids


class Tokenizer:
    cls_token_id = 2

    def __call__(self, words, **kwargs):
        # Split the middle word into two pieces to exercise alignment too.
        ids = [None] + [w for w in range(len(words)) for _ in range(2 if w == 1 else 1)] + [None]
        pieces = [2] + [4 + w for w in ids[1:-1]] + [3]
        enc = Encoding(input_ids=torch.tensor([pieces]), attention_mask=torch.ones(1, len(pieces), dtype=torch.long))
        enc.ids = ids
        return enc


@pytest.fixture
def tiny_bert(monkeypatch):
    torch.manual_seed(0)
    cfg = transformers.BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
        output_attentions=True, attn_implementation='eager')
    model = transformers.BertModel(cfg).eval()
    tok = Tokenizer()
    monkeypatch.setattr(attention, 'load_model', lambda *a, **kw: (tok, model))
    return tok, model


def test_real_forward_and_all_three_experiment_routes(tiny_bert):
    tok, model = tiny_bert
    data = synthetic_data()
    mats = attention.all_head_matrices([t.word for t in data[0][0][1:]], tok, model)
    assert mats.shape == (2, 2, 4, 4)
    assert np.isfinite(mats).all()
    args = SimpleNamespace(model_cache='unused', allow_download=False)
    base = dict(model='tiny_random_test_only', revision='none')
    mean = attention_experiment(data, data, dict(base, kind='attention', layers=[0, 1]), args)
    assert set(mean) == {'0', '1'}
    heads = attention_experiment(data, data, dict(base, kind='head-sweep', train_subset=2), args)
    assert heads['train_selection_sentences'] == 2
    assert 'test-selected' in heads['oracle_status']
    combine = attention_experiment(data, data, dict(base, kind='combine', train_subset=2,
        logistic_regression=dict(max_iter=2000, C=1., class_weight='balanced', random_state=0)), args)
    assert combine['features'] == 8
    assert combine['positive_edges'] == 6
    assert set(combine['macro_uas']) == {'max_all_heads', 'mean_all_heads', 'learned_combiner'}


def test_overlength_is_explicit(tiny_bert):
    tok, model = tiny_bert
    with pytest.raises(ValueError, match='exceeds model limit'):
        attention.all_head_matrices(['word'] * 40, tok, model)
