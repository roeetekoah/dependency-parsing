"""Explicit local corpus access. Importing the package never downloads data."""
import hashlib
from pathlib import Path


def load_data(test_fraction=0.1, data_dir=None):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if data_dir is None:
        raise ValueError("provide an NLTK data directory explicitly")
    root = Path(data_dir) / 'corpora' / 'dependency_treebank'
    if not root.is_dir():
        raise FileNotFoundError(f"missing corpus directory: {root}")
    import nltk
    from nltk.corpus.reader import DependencyCorpusReader
    from .perceptron import Token
    # NLTK's path security validates reader paths against its registered roots.
    corpus_base = str(Path(data_dir).resolve())
    if corpus_base not in nltk.data.path:
        nltk.data.path.insert(0, corpus_base)
    reader = DependencyCorpusReader(str(root), r'.*\.dp', encoding='ascii')
    data = []
    for graph in reader.parsed_sents():
        tokens, gold = [Token('ROOT', 'ROOT')], {}
        for address in sorted(graph.nodes):
            node = graph.nodes[address]
            if node['word'] is None:
                continue
            if address != len(tokens) or node['head'] is None:
                raise ValueError("corpus must have contiguous, headed tokens")
            tokens.append(Token(node['word'], node['tag']))
            gold[address] = node['head']
        if not gold:
            raise ValueError("empty sentence in corpus")
        data.append((tokens, gold))
    split = int(len(data) * (1 - test_fraction))
    if split == 0 or split == len(data):
        raise ValueError("split must contain both training and test sentences")
    return data[:split], data[split:]


def corpus_fingerprint(data_dir):
    """Hash file names and bytes in canonical order, without publishing text."""
    root = Path(data_dir) / 'corpora' / 'dependency_treebank'
    digest = hashlib.sha256()
    for p in sorted(root.glob('*.dp')):
        digest.update(p.name.encode('utf-8') + b'\0' + p.read_bytes() + b'\0')
    return digest.hexdigest()


def synthetic_data():
    """Tiny authored fixtures, not copied from a corpus; not a benchmark."""
    from .perceptron import Token
    return [([Token('ROOT', 'ROOT'), Token(a, 'NNS'), Token(b, 'VBP'),
              Token(c, 'NN')], {1: 2, 2: 0, 3: 2})
            for a, b, c in [('models', 'reveal', 'structure'),
                            ('graphs', 'encode', 'syntax'),
                            ('tests', 'check', 'inference')]]
