from pathlib import Path, PosixPath
from typing import Union, Optional

import torch
from torch import nn

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)

from .models import (
    BertForSequenceClassificationMeanPool,
    BertMeanPoolConfig,
    ModernBertForSequenceClassificationMeanPool,
    ModernBertMeanPoolConfig,
    RobertaForSequenceClassificationMeanPool,
    RobertaMeanPoolConfig,
)
from .nlp import DNABERTTokenizer


RobertaSettings = dict(
    padding_side='left'
)

ModernBertSettings = dict(
    padding_side='left'
)

DnabertSettings = dict(
    k=6,
    do_lower_case=False,
    padding_side='right'
)


MODELS = {
    "roberta-lm": (
        RobertaConfig,
        RobertaTokenizerFast,
        RobertaForMaskedLM,
        RobertaSettings
    ),
    "roberta-pred": (
        RobertaConfig,
        RobertaTokenizerFast,
        RobertaForSequenceClassification,
        RobertaSettings
    ),
    "roberta-pred-mean-pool": (
        RobertaMeanPoolConfig,
        RobertaTokenizerFast,
        RobertaForSequenceClassificationMeanPool,
        RobertaSettings
    ),
    "modernbert-lm": (
        ModernBertConfig,
        PreTrainedTokenizerFast,
        ModernBertForMaskedLM,
        ModernBertSettings
    ),
    "modernbert-pred": (
        ModernBertConfig,
        PreTrainedTokenizerFast,
        ModernBertForSequenceClassification,
        ModernBertSettings
    ),
    "modernbert-pred-mean-pool": (
        ModernBertMeanPoolConfig,
        PreTrainedTokenizerFast,
        ModernBertForSequenceClassificationMeanPool,
        ModernBertSettings
    ),
    "dnabert-lm": (
        BertConfig,
        DNABERTTokenizer,
        BertForMaskedLM,
        DnabertSettings
    ),
    "dnabert-pred": (
        BertConfig,
        DNABERTTokenizer,
        BertForSequenceClassification,
        DnabertSettings
    ),
    "dnabert-pred-mean-pool": (
        BertMeanPoolConfig,
        DNABERTTokenizer,
        BertForSequenceClassificationMeanPool,
        DnabertSettings
    )
}


def _is_local_checkpoint(path: Union[str, PosixPath]) -> bool:
    """Return whether ``path`` resolves to a local checkpoint directory."""
    return Path(str(path)).is_dir()


def _validate_local_checkpoint(path: Union[str, PosixPath]):
    """Fail early when a local checkpoint path is incomplete."""
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Pretrained checkpoint directory does not exist: {checkpoint}"
        )
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint is missing config.json: {checkpoint}"
        )
    weight_files = (
        list(checkpoint.glob("*.safetensors"))
        + list(checkpoint.glob("*.bin"))
        + list(checkpoint.glob("*.safetensors.index.json"))
        + list(checkpoint.glob("*.bin.index.json"))
    )
    if not weight_files:
        raise FileNotFoundError(
            f"Pretrained checkpoint has no model weight file (*.safetensors or *.bin): "
            f"{checkpoint}"
        )


def _validate_tokenizer_compatibility(
    tokenizer,
    checkpoint_config,
    max_position_embeddings: int,
    checkpoint_path: Union[str, PosixPath],
):
    """Check that a tokenizer can consume the selected checkpoint unchanged."""
    checkpoint_vocab_size = getattr(checkpoint_config, "vocab_size", None)
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != len(tokenizer):
        raise ValueError(
            "Tokenizer/checkpoint vocabulary mismatch: "
            f"tokenizer has {len(tokenizer)} entries but {checkpoint_path} expects "
            f"{checkpoint_vocab_size}. Do not resize or retrain the tokenizer for "
            "continued MLM."
        )

    checkpoint_max_position_embeddings = getattr(
        checkpoint_config, "max_position_embeddings", None
    )
    if (
        checkpoint_max_position_embeddings is not None
        and checkpoint_max_position_embeddings != max_position_embeddings
    ):
        raise ValueError(
            "Tokenizer/model sequence-length mismatch: "
            f"loader requested {max_position_embeddings} positions but "
            f"{checkpoint_path} contains {checkpoint_max_position_embeddings}."
        )

    token_ids = {
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "cls_token_id": getattr(tokenizer, "cls_token_id", None),
        "sep_token_id": getattr(tokenizer, "sep_token_id", None),
    }
    for name, tokenizer_id in token_ids.items():
        checkpoint_id = getattr(checkpoint_config, name, None)
        if (
            tokenizer_id is not None
            and checkpoint_id is not None
            and tokenizer_id != checkpoint_id
        ):
            raise ValueError(
                f"Tokenizer/checkpoint special-token mismatch for {name}: "
                f"tokenizer={tokenizer_id}, checkpoint={checkpoint_id}."
            )


def _expected_missing_key(model_name: str, key: str) -> bool:
    """Return whether a missing key is intentional for this model transition."""
    if key.endswith("position_ids") or ".position_ids" in key:
        return True
    # A language-model checkpoint intentionally has no downstream regression
    # head.  The base encoder must still be loaded and is checked separately.
    return model_name.startswith(("roberta-pred", "modernbert-pred", "dnabert-pred")) and key.startswith(
        "classifier."
    )


def _expected_unexpected_key(model_name: str, key: str) -> bool:
    """Return whether an unexpected key is intentional for this model transition."""
    # The regression model is initialized from an MLM checkpoint, so the MLM
    # prediction head is not part of the downstream architecture.
    return model_name.startswith(("roberta-pred", "modernbert-pred", "dnabert-pred")) and key.startswith(
        ("lm_head.", "head.", "cls.", "decoder.")
    )


def load_model(
    model_name: str,
    tokenizer_dir: Union[str, PosixPath],
    max_tokenized_len: int = 254,
    pretrained_model: Union[str, PosixPath] = None,
    k: Optional[int] = None,
    do_lower_case: Optional[bool] = None,
    padding_side: Optional[str] = 'left',
    **config_settings
) -> tuple:
    """Load specified model, config, and tokenizer.

    Args:
        model_name (str): Name of model. Acceptable options are
            - 'roberta-lm',
            - 'roberta-pred',
            - 'roberta-pred-mean-pool'
            - 'modernbert-lm',
            - 'modernbert-pred',
            - 'modernbert-pred-mean-pool'
            - 'dnabert-lm'
            - 'dnabert-pred'
            - 'dnabert-pred-mean-pool'
        tokenizer_dir (Union[str, PosixPath]): Directory containing tokenizer
            files: merges.txt and vocab.txt (RoBERTa) or a fast tokenizer
            (tokenizer.json) directory (ModernBERT).
        max_tokenized_len (int, optional): Maximum tokenized length,
            not including SOS and EOS. Defaults to 254.
        pretrained_model (Union[str, PosixPath], optional): Path to saved
            pretrained transformer model. Defaults to None.
        k (Optional[int], optional): Size of kmers (for DNABERT model).
            Defaults to 6.
        do_lower_case (bool, optional): Whether to convert all inputs to
            lower case. Defaults to None.
        padding_side (str, optional): Which side to pad on.
            Defaults to 'left'.

    Returns:
        tuple: config_obj, tokenizer, model
    """
    config_settings = config_settings or {}

    max_position_embeddings = max_tokenized_len + 2

    config_class, tokenizer_class, model_class, tokenizer_settings = MODELS[
        model_name
    ]

    kwargs = dict(
        max_len=max_tokenized_len,
        truncate=True,
        padding="max_length",
        **tokenizer_settings
    )

    if k is not None:
        kwargs.update(dict(k=k))

    if do_lower_case is not None:
        kwargs.update(dict(do_lower_case=do_lower_case))

    if padding_side is not None:
        kwargs.update(dict(padding_side=padding_side))

    tokenizer = tokenizer_class.from_pretrained(
        str(tokenizer_dir),
        **kwargs
    )

    # Cap model_max_length:
    # - ModernBERT needs this to avoid int(1e30) overflow.
    # - RoBERTa/BERT need this to avoid OOB position-embedding gather
    #   (position ids can reach num_tokens+1).
    if model_name.startswith("modernbert"):
        tokenizer.model_max_length = max_position_embeddings
    else:
        tokenizer.model_max_length = max_tokenized_len

    name_or_path = str(pretrained_model) or ''

    config_obj = config_class(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_position_embeddings,
        name_or_path=name_or_path,
        output_hidden_states=True,
        **config_settings
    )

    # ModernBERT can auto-compile layers with torch.compile when Triton
    # is available. Disable it for predictable/simple behavior in training.
    if model_name.startswith("modernbert") and hasattr(
        config_obj, "reference_compile"
    ):
        config_obj.reference_compile = False

    if pretrained_model:
        print(f"Loading from pretrained model {pretrained_model}")

        if _is_local_checkpoint(pretrained_model):
            _validate_local_checkpoint(pretrained_model)

        # Inspect the checkpoint's own architecture metadata before loading
        # weights into the configuration assembled from the repo settings.
        # This prevents an accidental embedding resize from hiding a tokenizer
        # mismatch during a continued-MLM run.
        checkpoint_config = config_class.from_pretrained(str(pretrained_model))
        _validate_tokenizer_compatibility(
            tokenizer,
            checkpoint_config,
            max_position_embeddings,
            pretrained_model,
        )

        loaded = model_class.from_pretrained(
            str(pretrained_model),
            config=config_obj,
            output_loading_info=True,
            _fast_init=False,
        )

        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError(
                "Transformers did not return loading information for the "
                f"pretrained checkpoint {pretrained_model}; refusing to assume "
                "that weights were loaded."
            )
        model, loading_info = loaded

        missing_keys = list(loading_info.get("missing_keys", []))
        unexpected_keys = list(loading_info.get("unexpected_keys", []))
        mismatched_keys = list(loading_info.get("mismatched_keys", []))
        relevant_missing = [
            key for key in missing_keys if not _expected_missing_key(model_name, key)
        ]
        relevant_unexpected = [
            key
            for key in unexpected_keys
            if not _expected_unexpected_key(model_name, key)
        ]
        if mismatched_keys:
            raise RuntimeError(
                f"Pretrained checkpoint has mismatched tensor shapes for "
                f"{model_name}: {mismatched_keys}"
            )
        if relevant_missing or relevant_unexpected:
            raise RuntimeError(
                f"Pretrained checkpoint did not match {model_name}. "
                f"Missing keys: {relevant_missing}; "
                f"unexpected keys: {relevant_unexpected}"
            )

        base_prefix = getattr(model, "base_model_prefix", None)
        base_parameter_names = [
            name
            for name, _ in model.named_parameters()
            if base_prefix
            and (name == base_prefix or name.startswith(f"{base_prefix}."))
        ]
        loaded_base_names = [
            name for name in base_parameter_names if name not in missing_keys
        ]
        if not base_parameter_names or not loaded_base_names:
            raise RuntimeError(
                f"No {model_name} base-encoder weights were loaded from "
                f"{pretrained_model}; refusing to train from a random base."
            )
        if len(loaded_base_names) != len(base_parameter_names):
            missing_base = [
                name for name in base_parameter_names if name in missing_keys
            ]
            raise RuntimeError(
                f"Only {len(loaded_base_names)}/{len(base_parameter_names)} "
                f"base-encoder parameters loaded from {pretrained_model}; "
                f"missing base keys: {missing_base}"
            )
        print(
            f"Verified pretrained load: {len(loaded_base_names):,} base parameter "
            f"tensors loaded from {pretrained_model}"
        )

        # Explicitly verify that the prediction/regression head exists and
        # contains valid initialized parameters.
        if hasattr(model, "classifier") and model.classifier is not None:
            head_is_nonfinite = any(
                not torch.isfinite(parameter.detach()).all()
                for parameter in model.classifier.parameters()
            )

            if head_is_nonfinite:
                print("Warning: reinitializing non-finite prediction head")

                for module in model.classifier.modules():
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()

            # Confirm the classifier is valid after loading/initialization.
            if any(
                not torch.isfinite(parameter.detach()).all()
                for parameter in model.classifier.parameters()
            ):
                raise RuntimeError(
                    "Prediction head contains non-finite parameters "
                    "after initialization"
                )
        elif model_name.endswith("-lm") and not (
            hasattr(model, "lm_head") or hasattr(model, "decoder")
        ):
            raise RuntimeError(
                f"Expected language-model '{model_name}' to expose an MLM head, "
                "but none was found."
            )
        elif not model_name.endswith("-lm"):
            raise RuntimeError(
                f"Expected model '{model_name}' to have a "
                "classifier/prediction head, but no classifier was found."
            )

    else:
        print("Loading untrained model")
        model = model_class(config=config_obj)

    model.resize_token_embeddings(len(tokenizer))

    embedding_count = model.get_input_embeddings().num_embeddings
    if embedding_count != len(tokenizer):
        raise RuntimeError(
            f"Model/tokenizer vocabulary mismatch after loading: model has "
            f"{embedding_count} embeddings, tokenizer has {len(tokenizer)}."
        )

    return config_obj, tokenizer, model
