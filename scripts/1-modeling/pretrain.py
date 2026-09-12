"""
Pretraining on masked language model task.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr


DATA_DIR = config.data_final / "transformer" / "seq"
DEFAULT_MODEL = "roberta-lm"


def _check_text_file(path: Path, label: str) -> int:
    """Validate a line-delimited sequence file and return non-empty line count."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} file is empty: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    if count == 0:
        raise ValueError(f"{label} file contains no non-empty sequences: {path}")
    return count


def _mlm_forward_smoke_test(model, tokenizer):
    """Run one cheap forward pass before allocating the Trainer."""
    model.eval()
    inputs = tokenizer(
        "ACGTACGTACGTTTTAAACCCGGG",
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    if outputs.logits.ndim != 3 or not torch.isfinite(outputs.logits).all():
        raise RuntimeError("MLM forward smoke test produced invalid logits")
    print(f"MLM forward smoke test logits shape: {tuple(outputs.logits.shape)}")
    model.train()


def main():
    pretrain_defaults = config.settings["training"]["pretrain"]
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data="all_seqs_train.txt",
        test_data="all_seqs_test.txt",
        output_dir=config.model_output_dir(DEFAULT_MODEL, "language-model"),
        tokenizer_dir=config.tokenizer_dir_for_model(DEFAULT_MODEL),
        model_name=DEFAULT_MODEL,
        pretrained_model=None,
        learning_rate=pretrain_defaults.get("learning_rate"),
        num_train_epochs=pretrain_defaults.get("num_train_epochs"),
        optimizer=pretrain_defaults.get("optimizer"),
        precision=(
            "bf16"
            if pretrain_defaults.get("bf16", False)
            else "fp16"
            if pretrain_defaults.get("fp16", False)
            else "no"
        ),
    )

    # Resolve model-specific defaults after parsing so selecting ModernBERT on
    # the CLI cannot silently reuse the RoBERTa output/tokenizer locations.
    if "--output-dir" not in sys.argv:
        args.output_dir = config.model_output_dir(args.model_name, "language-model")
    if "--tokenizer-dir" not in sys.argv:
        args.tokenizer_dir = config.tokenizer_dir_for_model(args.model_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved MLM train data: {args.train_data}")
    print(f"Resolved MLM eval data: {args.test_data}")
    print(f"Resolved tokenizer directory: {args.tokenizer_dir}")
    print(f"Resolved output directory: {args.output_dir}")
    if args.pretrained_model:
        print(f"Selected pretrained checkpoint: {args.pretrained_model}")
    else:
        print("Selected pretrained checkpoint: none (explicit scratch pretraining)")

    train_count = _check_text_file(args.train_data, "MLM training")
    test_count = _check_text_file(args.test_data, "MLM evaluation")
    print(f"Non-empty MLM train sequences: {train_count:,}")
    print(f"Non-empty MLM eval sequences: {test_count:,}")

    print(args)

    settings = utils.get_model_settings(config.settings, args, args.model_name)

    config_obj, tokenizer, model = tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        **settings,
    )

    trainable_params = utils.count_model_parameters(model, trainable_only=True)
    total_params = utils.count_model_parameters(model, trainable_only=False)
    print(
        f"Loaded {args.model_name} model with {trainable_params:,} trainable / "
        f"{total_params:,} total parameters"
    )
    _mlm_forward_smoke_test(model, tokenizer)

    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        test_data=args.test_data,
        file_type="text",
        seq_key="text",
        n_workers=args.n_workers or min(os.cpu_count() or 1, 8),
    )
    dataset_train = datasets["train"]
    dataset_test = datasets["test"]
    print(f"Loaded training data with {len(dataset_train):,} examples")
    data_collator = dataio.load_data_collator(
        "language-model",
        tokenizer=tokenizer,
        mlm_prob=pretrain_defaults.get("mlm_prob", 0.15),
    )

    training_settings = dict(pretrain_defaults)
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs
    if args.optimizer is not None:
        training_settings["optimizer"] = args.optimizer
    # ``precision`` is a CLI convenience; TrainingArguments uses fp16/bf16.
    training_settings["fp16"] = args.precision == "fp16"
    training_settings["bf16"] = args.precision == "bf16"
    print(f"Effective MLM training settings: {training_settings}")

    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_test,
        args.output_dir,
        **training_settings,
    )

    print(
        f"Starting training on {torch.cuda.device_count()} GPUs"
        if "COLAB_TPU_ADDR" not in os.environ
        else "Starting TPU training"
    )
    training.do_training(trainer, args, args.output_dir)

    print("Saving model")

    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
