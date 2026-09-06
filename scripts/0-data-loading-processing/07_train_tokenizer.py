"""Training byte-level BPE tokenizers for RoBERTa or ModernBERT base models.

By default we take 1/20 of the corpus (``total // 20``) -- the same volume the
original florabert-2 20-iteration training effectively used per chunk. Override
with ``--max-sequences``.
"""
import argparse
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module.florabert import config


SETTINGS = config.settings["tokenizer"]

SPECIAL_TOKENS = {
    # RoBERTa-style special tokens (order determines token ids: <s>=0, </s>=1, ...)
    "roberta": ["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
    # ModernBERT-style special tokens (order determines token ids: [CLS]=0, [SEP]=1, ...)
    "modernbert": ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"],
}


def batch_iterator(path: Path, batch_size: int = 10_000, max_sequences: int = None):
    """Yield non-empty lines of `path` in batches, lazily, up to `max_sequences`.

    Batching avoids materialising the whole file in Python memory; the cap
    bounds the trainer's internal word-frequency table (the actual RAM hog).
    """
    n = 0
    with open(str(path), "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            n += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if max_sequences is not None and n >= max_sequences:
                break
        if batch:
            yield batch


def count_sequences(path: Path) -> int:
    """Count non-empty lines without loading the file into memory."""
    n = 0
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer for RoBERTa or ModernBERT."
    )
    parser.add_argument(
        "--model",
        choices=["roberta", "modernbert"],
        default="roberta",
        help="Which special-token convention to use. 'modernbert' also saves the "
        "tokenizer as a fast-tokenizer directory (tokenizer.json) required by "
        "PreTrainedTokenizerFast (ModernBERT has no dedicated tokenizer class).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of sequences fed to the trainer in a single "
        "train_from_iterator call. The trainer keeps one entry per unique "
        "sequence in memory, so this bounds peak RAM (roughly 3-4 GB per 1M "
        "sequences). Default: total // 20 (mirrors the original 20-iteration "
        "training chunk size).",
    )
    args = parser.parse_args()

    if args.model == "modernbert":
        TOKENIZER_DIR = config.modernbert_tokenizer_dir
    else:
        TOKENIZER_DIR = config.roberta_tokenizer_dir
    if not TOKENIZER_DIR.exists():
        TOKENIZER_DIR.mkdir()

    DATA_DIR = config.data_final / "transformer" / "seq"
    sample_data = DATA_DIR / "all_seqs_train_sample.txt"
    full_data = DATA_DIR / "all_seqs_train.txt"
    if sample_data.exists():
        TRAIN_DATA = sample_data
    elif full_data.exists():
        print(f"NOTE: {sample_data.name} not found, using {full_data.name}")
        TRAIN_DATA = full_data
    else:
        raise FileNotFoundError(
            f"No training sequences found under {DATA_DIR}. Expected either "
            f"{sample_data.name} or {full_data.name}."
        )
    special_tokens = SPECIAL_TOKENS[args.model]
    vocab_size = SETTINGS["vocab_size"]

    max_sequences = args.max_sequences
    if max_sequences is None:
        total = count_sequences(TRAIN_DATA)
        max_sequences = max(total // 20, 1)
        print(f"NOTE: --max-sequences not set; defaulting to 1/20 of corpus "
              f"({max_sequences:,} of {total:,} sequences) to match the original "
              f"20-iteration training chunk size.")

    print(f"Training tokenizer ({args.model}), vocab_size={vocab_size}, "
          f"max_sequences={max_sequences}")
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        iterator=batch_iterator(TRAIN_DATA, max_sequences=max_sequences),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    if args.model == "modernbert":
        print("Saving ModernBERT fast tokenizer")
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )
        max_tokenized_len = config.settings["models"]["modernbert-base"].get(
            "max_tokenized_len", 256
        )
        fast_tokenizer.model_max_length = max_tokenized_len + 2
        fast_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    else:
        print("Saving tokenizer")
        tokenizer.save_model(str(TOKENIZER_DIR))

    print("Done")


if __name__ == "__main__":
    main()