import pytest

import torch

from torch.optim import Adam

from florabert import config, training, utils
from florabert import transformers as tr


# Helper functions
def load_roberta_model(**kwargs):
    return tr.load_model(
        "roberta-pred-mean-pool", config.models / "byte-level-bpe-tokenizer", **kwargs
    )


def make_modernbert_tokenizer(tmp_path):
    """Train a tiny byte-level BPE tokenizer with ModernBERT special tokens and
    save it as a fast-tokenizer directory (tokenizer.json), i.e. the format
    produced by scripts/0-data-loading-processing/07_train_tokenizer.py
    (`--model modernbert`)."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["ACGT" * 200, "TGCA" * 200, "AAAA" * 400, "NNNN" * 100],
        vocab_size=300,
        special_tokens=["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"],
    )
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
    fast_tokenizer.save_pretrained(str(tmp_path))
    return tmp_path


def load_modernbert_model(model_name="modernbert-pred-mean-pool", **kwargs):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer_dir = make_modernbert_tokenizer(__import__("pathlib").Path(tmp_dir))
        return tr.load_model(
            model_name,
            tokenizer_dir,
            max_tokenized_len=128,
            padding_side="left",
            **kwargs,
        )


# Tests
def test_roberta_mean_pool_load_new():
    try:
        load_roberta_model()
    except:
        raise Exception(
            "Failed to load new RobertaForSequenceClassificationMeanPool model"
        )


def test_roberta_mean_pool_load_pretrained():
    try:
        load_roberta_model(
            pretrained_model=config.models / "transformer" / "language-model"
        )
    except:
        raise Exception(
            "Failed to load pretrained RobertaForSequenceClassificationMeanPool model"
        )


def test_modernbert_lm_load():
    config_obj, tokenizer, model = load_modernbert_model(
        "modernbert-lm", num_labels=8
    )
    assert model.config.model_type == "modernbert"
    assert model.base_model_prefix == "model"
    assert len(tokenizer) == config_obj.vocab_size


def test_modernbert_lm_forward():
    config_obj, tokenizer, model = load_modernbert_model("modernbert-lm")
    inputs = tokenizer("ACGTACGTACGTTTTAAACCCGGG", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    # logits: (batch, seq_len, vocab_size)
    assert outputs.logits.shape[0] == inputs["input_ids"].shape[0]
    assert outputs.logits.shape[-1] == len(tokenizer)
    assert outputs.loss is None


def test_modernbert_mean_pool_load_new():
    load_modernbert_model("modernbert-pred-mean-pool", num_labels=8)


def test_modernbert_mean_pool_forward():
    config_obj, tokenizer, model = load_modernbert_model(
        "modernbert-pred-mean-pool", num_labels=8
    )
    inputs = tokenizer(["ACGTACGTACGTTTTAAACCCGGG", "TTTTGGGGCCCCAAAA"], return_tensors="pt", padding="max_length", max_length=128)
    labels = torch.randn(2, 8)
    model.train()
    outputs = model(**inputs, labels=labels)
    assert outputs.logits.shape == (2, 8)
    assert outputs.loss is not None


def test_modernbert_freeze_base():
    model = load_modernbert_model("modernbert-pred-mean-pool", num_labels=8)[2]
    base = utils.get_model_base(model)
    assert type(base).__name__ == "ModernBertModel"
    utils.freeze_base(model)
    all_frozen = all(not p.requires_grad for p in base.parameters())
    assert all_frozen, "Base encoder should be fully frozen"


def test_get_lamb_optimizer():
    _, _, model = load_roberta_model()
    optimizer = training._get_optimizer("lamb", model)
    assert optimizer is not None, "Failed to load optimizer"


def test_linear_scheduler():
    tr._get_scheduler("linear", Adam(), 10000, num_warmup_steps=500)


def test_delay_scheduler():
    tr._get_scheduler(
        "delay", Adam(), 10000, num_warmup_steps=500, num_param_groups=4, delay_size=400
    )


def test_make_trainer_simple():
    pass


def test_make_trainer_delay():
    pass


def test_load_datasets():
    pass


def test_convert_str_to_list():
    data = ["[32.0, 430.5]", "[20.0, 0.01]", "[-1419, 4]"]
    lists = tr.convert_str_to_list(data)

    assert type(lists) == list, "Result is not a list"
    assert all([type(li) == list] for li in lists), "Inner lists are not lists"
    assert all(
        [all([type(d) == float for d in li]) for li in lists]
    ), "Elements are not float"
