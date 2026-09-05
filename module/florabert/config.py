from pathlib import Path
import yaml
import random

import numpy as np
import torch


root = Path(__file__).parent.parent.parent
data = root / 'data'
models = root / 'models'
notebooks = root / 'notebooks'
scripts = root / 'scripts'
output = root / 'output'
docs = root / 'docs'

# Data specific paths
data_raw = data / 'raw'
data_processed = data / 'processed'
data_final = data / 'final'

# Location of tools
libs = root / 'libs'
samtools = libs / 'samtools'
bedtools = libs / 'bedtools'
dnabert = root / 'DNABERT'

# Locations of specific files
bpe_tokenizer = data_final / 'tokenizer' / 'maize_bpe_full.tokenizer.json'

# Tokenizer directories (for RoBERTa and ModernBERT byte-level-BPE tokenizers)
roberta_tokenizer_dir = models / 'byte-level-bpe-tokenizer'
modernbert_tokenizer_dir = models / 'modernbert-byte-level-bpe-tokenizer'


def tokenizer_dir_for_model(model_name: str) -> Path:
    """Return the tokenizer directory appropriate for `model_name`.

    ModernBERT uses a separate byte-level-BPE tokenizer trained with
    ModernBERT-style special tokens ([CLS]/[SEP]/[PAD]/[UNK]/[MASK]) and saved as
    a fast-tokenizer (`tokenizer.json`) directory. Everything else uses the
    original RoBERTa byte-level-BPE tokenizer.
    """
    base = model_name.split('-')[0]
    if base == 'modernbert':
        return modernbert_tokenizer_dir
    return roberta_tokenizer_dir


def model_output_dir(model_name: str, stage: str) -> Path:
    """Return the model output directory for `stage` ('language-model' or
    'prediction-model'), keeping ModernBERT checkpoints separate from the
    original RoBERTa ones under models/transformer/<stage>."""
    base = model_name.split('-')[0]
    if base == 'modernbert':
        return models / 'transformer' / f'{stage}-{base}'
    return models / 'transformer' / stage

# Loading settings
settings = yaml.full_load((root / 'config.yaml').open('r'))

# Setting random seeds across the whole project
random_seed = settings['random_seed']
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def reload_settings():
    global settings
    settings = yaml.full_load((root / 'config.yaml').open('r'))


# New (NAM)
# plant embryo and shoot aren't available for all cultivars
tissues = [
    'tassel', 
    'base', 
    'anther', 
    'middle', 
    'ear', 
    'shoot', 
    'tip', 
    'root' 
]

# OLD
# tissues = [
#     'anther',
#     'ear',
#     'embryo',
#     'endosperm',
#     'leaf',
#     'leafbase',
#     'leaftip',
#     'root',
#     'shoot',
#     'tassel'
# ]
