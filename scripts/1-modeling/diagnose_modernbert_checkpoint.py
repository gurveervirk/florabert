"""Diagnose ModernBERT checkpoint outputs before finetuning."""
import argparse
import sys
from pathlib import Path

import torch
from transformers import default_data_collator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module.florabert import config, dataio, transformers as tr, utils


def summarize(name, tensor):
    values = tensor.detach().float()
    print(
        f"{name}: shape={tuple(values.shape)} finite={bool(torch.isfinite(values).all())} "
        f"min={values.min().item():.6g} max={values.max().item():.6g} "
        f"absmax={values.abs().max().item():.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default=None)
    parser.add_argument(
        "--pretrained-model",
        default=str(config.model_output_dir("modernbert-lm", "language-model")),
    )
    parser.add_argument(
        "--data-dir",
        default=str(config.data_final / "transformer" / "genex" / "nam"),
    )
    parser.add_argument("--data", default="train.tsv")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_settings = utils.get_model_settings(config.settings, model_name="modernbert-lm")
    pred_settings = utils.get_model_settings(
        config.settings, model_name="modernbert-pred-mean-pool"
    )
    if args.attention:
        base_settings["attn_implementation"] = args.attention
        pred_settings["attn_implementation"] = args.attention

    print(f"device={device} attention={args.attention or 'configured default'}")
    print(f"checkpoint={args.pretrained_model}")

    _, tokenizer, _ = tr.load_model(
        "modernbert-lm",
        config.tokenizer_dir_for_model("modernbert-lm"),
        pretrained_model=args.pretrained_model,
        **base_settings,
    )
    datasets = dataio.load_datasets(
        tokenizer,
        Path(args.data_dir) / args.data,
        seq_key="sequence",
        file_type="csv",
        delimiter="\t",
        log_offset=1,
        transformation="log",
        shuffle=False,
    )
    row = datasets["train"].remove_columns(["sequence"])[0]
    batch = default_data_collator([row])
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "position_ids"}
    }

    print("\nDirect MLM model")
    _, _, mlm = tr.load_model(
        "modernbert-lm",
        config.tokenizer_dir_for_model("modernbert-lm"),
        pretrained_model=args.pretrained_model,
        **base_settings,
    )
    mlm.to(device).eval()
    with torch.no_grad():
        mlm_outputs = mlm(**inputs)
    summarize("mlm/logits", mlm_outputs.logits)
    summarize("mlm/last_hidden_state", mlm_outputs.hidden_states[-1])

    print("\nCustom prediction model")
    _, _, predictor = tr.load_model(
        "modernbert-pred-mean-pool",
        config.tokenizer_dir_for_model("modernbert-pred-mean-pool"),
        pretrained_model=args.pretrained_model,
        **pred_settings,
    )
    predictor.to(device).eval()
    with torch.no_grad():
        base_outputs = predictor.model(**inputs)
        pooled = predictor.classifier.embed(
            base_outputs.last_hidden_state,
            attention_mask=inputs["attention_mask"],
            input_ids=inputs["input_ids"],
        )
        prediction_outputs = predictor(**inputs)
    summarize("predictor/base_last_hidden_state", base_outputs.last_hidden_state)
    summarize("predictor/pooled_head_input", pooled)
    summarize("predictor/logits", prediction_outputs.logits)


if __name__ == "__main__":
    main()
