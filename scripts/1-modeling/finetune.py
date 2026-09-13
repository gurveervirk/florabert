"""
Fine-tuning the transformer model on the downstream gene expression prediction task
using accelerate for manual train/eval loops.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional for offline/local runs
    wandb = None

from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr


DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TRAIN_DATA = "train.tsv"
EVAL_DATA = "eval.tsv"
TEST_DATA = "test.tsv"
DEFAULT_MODEL = "roberta-pred-mean-pool"
PREPROCESSOR = None
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids", "labels")


def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )


def _head_diagnostics(model, optimizer):
    """Return prediction-head norms and optimizer-specific diagnostics.

    LAMB exposes useful trust-ratio state, but StableAdamW does not. Keep the
    LAMB fields for historical runs while avoiding misleading null fields for
    other optimizers.
    """
    head = model.classifier
    output_layer = head.out_proj
    weight = output_layer.weight
    diagnostics = {
        "head/weight_norm": float(weight.detach().float().norm().cpu()),
        "head/bias_norm": float(output_layer.bias.detach().float().norm().cpu()),
    }

    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    if base_optimizer.__class__.__name__.lower() == "lamb":
        state = base_optimizer.state.get(weight, {})

        def scalar(name):
            value = state.get(name)
            return float(value.detach().float().cpu()) if value is not None else None

        diagnostics.update(
            {
                "lamb/weight_norm": scalar("weight_norm"),
                "lamb/adam_norm": scalar("adam_norm"),
                "lamb/trust_ratio": scalar("trust_ratio"),
            }
        )

    return diagnostics


def _first_nonfinite_tensor(tensors):
    """Return the first named tensor containing NaN or Inf values."""
    for name, tensor in tensors:
        if tensor is not None and not torch.isfinite(tensor.detach()).all():
            return name
    return None


def _assert_finite_model(model, stage):
    name = _first_nonfinite_tensor(model.named_parameters())
    if name is not None:
        raise RuntimeError(f"Non-finite model parameter after {stage}: {name}")


def _tensor_summary(tensor):
    values = tensor.detach().float()
    return (
        f"min={values.min().item():.6g}, max={values.max().item():.6g}, "
        f"absmax={values.abs().max().item():.6g}"
    )


def _population_std(values):
    """Return a JSON-friendly population standard deviation."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values, ddof=0)) if values.size else float("nan")


def _metric_triplet(targets, predictions):
    """Compute the fixed regression metrics for one target vector."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    return {
        "mse": float(np.mean((targets - predictions) ** 2)),
        "r2": float(utils.compute_r2(targets, predictions)),
        "pearson_r2": float(utils.compute_pearson_r2(targets, predictions)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": _population_std(predictions),
        "target_mean": float(np.mean(targets)),
        "target_std": _population_std(targets),
    }


def _evaluation_metrics(labels, predictions):
    """Return overall and per-tissue metrics in transformed target space."""
    labels_np = labels.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()
    overall = _metric_triplet(labels_np.ravel(), predictions_np.ravel())
    per_tissue = {}
    for tissue_idx in range(labels_np.shape[1]):
        tissue = (
            config.tissues[tissue_idx]
            if tissue_idx < len(config.tissues)
            else f"tissue_{tissue_idx}"
        )
        per_tissue[tissue] = _metric_triplet(
            labels_np[:, tissue_idx], predictions_np[:, tissue_idx]
        )
    return {"overall": overall, "per_tissue": per_tissue}


def _load_previous_best(metrics_path):
    """Recover the best validation MSE when resuming a regression run."""
    if not metrics_path.is_file():
        return float("inf"), None
    best_mse = float("inf")
    best_epoch = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            overall = record.get("overall", {})
            mse = overall.get("mse")
            if mse is not None and mse < best_mse:
                best_mse = float(mse)
                best_epoch = record.get("epoch")
    return best_mse, best_epoch


def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data=TRAIN_DATA,
        eval_data=EVAL_DATA,
        test_data=TEST_DATA,
        output_dir=config.model_output_dir(DEFAULT_MODEL, "prediction-model"),
        pretrained_model=config.model_output_dir(DEFAULT_MODEL, "language-model"),
        tokenizer_dir=config.tokenizer_dir_for_model(DEFAULT_MODEL),
        model_name=DEFAULT_MODEL,
        log_offset=config.settings["training"]["finetune"].get("log_offset", 0.001),
        preprocessor=PREPROCESSOR,
        transformation=config.settings["training"]["finetune"]["transformation"],
        learning_rate=config.settings["training"]["finetune"]["learning_rate"],
        num_train_epochs=config.settings["training"]["finetune"]["num_train_epochs"],
        optimizer=config.settings["training"]["finetune"].get("optimizer"),
        precision=config.settings["training"]["finetune"].get("precision", "bf16"),
        hyperparam_search_metrics="mse",
        hyperparam_search_trials=10,
    )

    if "--output-dir" not in sys.argv:
        args.output_dir = config.model_output_dir(args.model_name, "prediction-model")
    if "--tokenizer-dir" not in sys.argv:
        args.tokenizer_dir = config.tokenizer_dir_for_model(args.model_name)
    if args.resume_from_checkpoint and "--pretrained-model" not in sys.argv:
        args.pretrained_model = args.resume_from_checkpoint
    elif "--pretrained-model" not in sys.argv:
        args.pretrained_model = config.model_output_dir(args.model_name, "language-model")

    if args.resume_from_checkpoint:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Regression resume checkpoint does not exist: "
                f"{args.resume_from_checkpoint}"
            )
        print(f"Selected regression resume checkpoint: {args.resume_from_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(args)
    settings = utils.get_model_settings(config.settings, args)

    print("Making model")
    config_obj, tokenizer, model = load_model(args, settings)
    _assert_finite_model(model, "model load")
    if args.freeze_base:
        print("Freezing base")
        utils.freeze_base(model)

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")

    print("Loading data")
    preprocessor = utils.load_pickle(args.preprocessor) if args.preprocessor else None
    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        eval_data=args.eval_data,
        test_data=args.test_data,
        seq_key="sequence",
        file_type="csv",
        delimiter="\t",
        log_offset=args.log_offset,
        preprocessor=preprocessor,
        filter_empty=args.filter_empty,
        tissue_subset=args.tissue_subset,
        threshold=args.threshold,
        transformation=args.transformation,
        discretize=(args.output_mode == "classification"),
        nshards=args.nshards,
        n_workers=args.n_workers or min(os.cpu_count() or 1, 8),
    )
    dataset_train = datasets["train"].remove_columns(["sequence"])
    dataset_eval = datasets["eval"].remove_columns(["sequence"])
    print(
        f"Loaded training data with {len(dataset_train)} examples and "
        f"validation data with {len(dataset_eval)} examples"
    )

    data_collator = dataio.load_data_collator("pred")
    training_settings = dict(config.settings["training"]["finetune"])
    # Keep the displayed and logged precision aligned with the Accelerator
    # argument; the legacy config flags may still contain fp16: true.
    training_settings["precision"] = args.precision
    training_settings["fp16"] = args.precision == "fp16"
    training_settings["bf16"] = args.precision == "bf16"
    debug_numerics = args.debug_numerics or training_settings.get("debug_numerics", False)
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs
    if args.optimizer is not None:
        training_settings["optimizer"] = args.optimizer
    if args.no_grad_clipping:
        training_settings.pop("max_grad_norm", None)
        print("Conventional gradient clipping disabled by --no-grad-clipping")
    print(training_settings)

    num_epochs = int(training_settings.get("num_train_epochs", 3))
    train_batch_size = training_settings.get("per_device_train_batch_size", 64)
    eval_batch_size = training_settings.get("per_device_eval_batch_size", 8)

    accelerator = Accelerator(mixed_precision=args.precision)
    selected_precision = accelerator.state.mixed_precision
    accelerator.print(f"Selected mixed precision: {selected_precision}")

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        dataset_eval,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    num_training_steps = int(
        np.ceil(len(dataset_train) / (train_batch_size * accelerator.num_processes))
        * num_epochs
    )
    optimizer, scheduler = training.make_optimizer_and_scheduler(
        model, training_settings, num_training_steps=num_training_steps
    )

    train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, scheduler
    )

    wandb_enabled = (
        wandb is not None
        and os.environ.get("WANDB_DISABLED", "").lower()
        not in {"1", "true", "yes"}
    )
    if accelerator.is_main_process and wandb_enabled:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "florabert"),
            config={
                **training_settings,
                "model_name": args.model_name,
                "transformation": args.transformation,
                # Record the effective Accelerator setting, not the legacy
                # TrainingArguments-style fp16 flag from config.yaml.
                "precision": selected_precision,
                "mixed_precision": selected_precision,
                "fp16": selected_precision == "fp16",
                "bf16": selected_precision == "bf16",
                "train_size": len(dataset_train),
                "eval_size": len(dataset_eval),
                "num_trainable_params": num_params,
                **training_settings,
            },
        )
    elif accelerator.is_main_process:
        print("W&B logging disabled (install wandb and unset WANDB_DISABLED to enable)")

    start_epoch = 0
    global_step = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    best_val_mse, best_epoch = _load_previous_best(metrics_path)
    if args.resume_from_checkpoint:
        state_path = args.resume_from_checkpoint / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(
                "Regression resume requires training_state.pt alongside the model "
                f"checkpoint: {state_path}"
            )
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        accelerator.print(
            f"Restored optimizer/scheduler state at epoch {start_epoch}, "
            f"global step {global_step}"
        )

    steps_per_epoch = int(
        np.ceil(len(dataset_train) / (train_batch_size * accelerator.num_processes))
    )
    num_training_steps = steps_per_epoch * num_epochs
    progress_bar = tqdm(
        range(global_step, num_training_steps),
        disable=not accelerator.is_local_main_process,
    )
    logging_steps = int(training_settings.get("logging_steps", 50))
    running_loss = 0.0
    accelerator.print("Starting training")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}
            if not torch.isfinite(inputs["labels"]).all():
                raise RuntimeError(f"Non-finite labels at training step {global_step}")
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss.detach()).all():
                raise RuntimeError(
                    f"Non-finite loss before backward at training step {global_step}: "
                    f"{loss}; labels({_tensor_summary(inputs['labels'])}); "
                    f"logits({_tensor_summary(outputs.logits)})"
                )
            if not torch.isfinite(outputs.logits.detach()).all():
                raise RuntimeError(
                    f"Non-finite logits before backward at training step {global_step}"
                )
            accelerator.backward(loss)
            grad_norm = None
            max_grad_norm = training_settings.get("max_grad_norm")
            if max_grad_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                ).item()
                if debug_numerics and not np.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at training step {global_step}: "
                        f"{grad_norm}"
                    )
            else:
                # Ensure fp16 gradients are unscaled before optional inspection.
                accelerator.unscale_gradients()
            if debug_numerics:
                bad_grad = _first_nonfinite_tensor(
                    (name, parameter.grad)
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                )
                if bad_grad is not None:
                    raise RuntimeError(
                        f"Non-finite gradient before optimizer step at training step "
                        f"{global_step}: {bad_grad}"
                    )
            optimizer.step()
            if debug_numerics:
                _assert_finite_model(model, f"optimizer step {global_step}")
            scheduler.step()
            progress_bar.update(1)

            running_loss += loss.detach().float().item()
            global_step += 1
            if global_step % logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                if accelerator.is_main_process:
                    logits = outputs.logits.detach().float()
                    log = {
                        "epoch": epoch + (global_step % max(steps_per_epoch, 1)) / max(steps_per_epoch, 1),
                        "loss": running_loss / logging_steps,
                        "learning_rate": lr,
                        "step": global_step,
                        "precision": selected_precision,
                        "train/logit_mean": float(logits.mean().cpu()),
                        "train/logit_std": float(logits.std().cpu()),
                    }
                    if grad_norm is not None:
                        log["grad_norm"] = grad_norm
                    log.update(_head_diagnostics(accelerator.unwrap_model(model), optimizer))
                    if wandb_enabled:
                        wandb.log(log)
                running_loss = 0.0

        model.eval()
        all_predictions = []
        all_labels = []
        for batch in eval_dataloader:
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}
            with torch.no_grad():
                outputs = model(**inputs)
            gathered_predictions, gathered_labels = accelerator.gather_for_metrics(
                (outputs.logits, labels)
            )
            all_predictions.append(gathered_predictions.detach().cpu())
            all_labels.append(gathered_labels.detach().cpu())

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        evaluation = _evaluation_metrics(all_labels, all_predictions)
        overall = evaluation["overall"]
        eval_mse = overall["mse"]
        eval_r2 = overall["r2"]
        accelerator.print(
            f"epoch {epoch + 1}: validation mse={eval_mse:.4f} "
            f"r2={eval_r2:.4f} pearson_r2={overall['pearson_r2']:.4f} "
            f"pred_mean={overall['prediction_mean']:.6f} "
            f"pred_std={overall['prediction_std']:.6f} "
            f"target_mean={overall['target_mean']:.6f} "
            f"target_std={overall['target_std']:.6f}"
        )
        for tissue, tissue_metrics in evaluation["per_tissue"].items():
            accelerator.print(
                f"  {tissue}: mse={tissue_metrics['mse']:.4f} "
                f"r2={tissue_metrics['r2']:.4f} "
                f"pearson_r2={tissue_metrics['pearson_r2']:.4f} "
                f"pred_mean={tissue_metrics['prediction_mean']:.6f} "
                f"pred_std={tissue_metrics['prediction_std']:.6f} "
                f"target_mean={tissue_metrics['target_mean']:.6f} "
                f"target_std={tissue_metrics['target_std']:.6f}"
            )
        if accelerator.is_main_process:
            with metrics_path.open("a", encoding="utf-8") as handle:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "overall": overall,
                        "per_tissue": evaluation["per_tissue"],
                    },
                    handle,
                )
                handle.write("\n")
            eval_log = _head_diagnostics(accelerator.unwrap_model(model), optimizer)
            eval_log.update(
                {
                    "epoch": epoch + 1,
                    "eval/mse": float(eval_mse),
                    "eval/r2": float(eval_r2),
                    "eval/pearson_r2": overall["pearson_r2"],
                    "eval/pred_mean": overall["prediction_mean"],
                    "eval/pred_std": overall["prediction_std"],
                    "eval/target_mean": overall["target_mean"],
                    "eval/target_std": overall["target_std"],
                }
            )
            if wandb_enabled:
                wandb.log(eval_log)

        unwrapped_model = accelerator.unwrap_model(model)
        epoch_dir = args.output_dir / f"epoch_{epoch}"
        accelerator.wait_for_everyone()
        unwrapped_model.save_pretrained(
            epoch_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                epoch_dir / "training_state.pt",
            )
        accelerator.wait_for_everyone()

        if eval_mse < best_val_mse:
            best_val_mse = eval_mse
            best_epoch = epoch + 1
            accelerator.wait_for_everyone()
            unwrapped_model.save_pretrained(
                args.output_dir / "best",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                with (args.output_dir / "best_metrics.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {
                            "epoch": best_epoch,
                            "validation": evaluation,
                            "checkpoint": str(args.output_dir / "best"),
                        },
                        handle,
                        indent=2,
                    )
                print(
                    f"New best validation checkpoint: {args.output_dir / 'best'} "
                    f"(epoch {best_epoch}, mse={best_val_mse:.6f})"
                )
            accelerator.wait_for_everyone()

    print("Saving model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir / "final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if best_epoch is None:
        raise RuntimeError(
            "No validation checkpoint was produced; refusing to report a final "
            "regression model as the best model."
        )
    accelerator.print(
        f"Best validation checkpoint: {args.output_dir / 'best'} "
        f"(epoch {best_epoch}, mse={best_val_mse:.6f})"
    )

    if accelerator.is_main_process and wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
