"""
Fine-tuning the transformer model on the downstream gene expression prediction task
using accelerate for manual train/eval loops.
"""
import os
import sys
sys.path.append('/kaggle/working/florabert')
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

import wandb

from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr
from module.florabert.utils import compute_r2, compute_mse


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
        log_offset=1,
        preprocessor=PREPROCESSOR,
        transformation="log10",
        hyperparam_search_metrics="mse",
        hyperparam_search_trials=10,
    )

    if "--output-dir" not in sys.argv:
        args.output_dir = config.model_output_dir(args.model_name, "prediction-model")
    if "--tokenizer-dir" not in sys.argv:
        args.tokenizer_dir = config.tokenizer_dir_for_model(args.model_name)
    if "--pretrained-model" not in sys.argv:
        args.pretrained_model = config.model_output_dir(args.model_name, "language-model")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(args)
    settings = utils.get_model_settings(config.settings, args)

    print("Making model")
    config_obj, tokenizer, model = load_model(args, settings)
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
    )
    dataset_train = datasets["train"].remove_columns(["sequence"])
    dataset_eval = datasets["eval"].remove_columns(["sequence"])
    print(f"Loaded training data with {len(dataset_train)} examples")

    data_collator = dataio.load_data_collator("pred")
    training_settings = config.settings["training"]["finetune"]
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs
    print(training_settings)

    num_epochs = int(training_settings.get("num_train_epochs", 3))
    train_batch_size = training_settings.get("per_device_train_batch_size", 64)
    eval_batch_size = training_settings.get("per_device_eval_batch_size", 8)

    accelerator = Accelerator(mixed_precision="fp16")

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

    if accelerator.is_main_process:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "florabert"),
            config={
                "model_name": args.model_name,
                "transformation": args.transformation,
                "train_size": len(dataset_train),
                "eval_size": len(dataset_eval),
                "num_trainable_params": num_params,
                **training_settings,
            },
        )

    progress_bar = tqdm(
        range(num_training_steps),
        disable=not accelerator.is_local_main_process,
    )
    steps_per_epoch = num_training_steps // num_epochs
    logging_steps = int(training_settings.get("logging_steps", 50))
    global_step = 0
    running_loss = 0.0
    accelerator.print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}
            outputs = model(**inputs)
            loss = outputs.loss
            accelerator.backward(loss)
            grad_norm = None
            if "max_grad_norm" in training_settings:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), training_settings["max_grad_norm"]
                ).item()
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)

            running_loss += loss.detach().float().item()
            global_step += 1
            if global_step % logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                if accelerator.is_main_process:
                    log = {
                        "epoch": epoch + (global_step % steps_per_epoch) / steps_per_epoch,
                        "loss": running_loss / logging_steps,
                        "learning_rate": lr,
                        "step": global_step,
                    }
                    if grad_norm is not None:
                        log["grad_norm"] = grad_norm
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
            all_predictions.append(accelerator.gather(outputs.logits).detach().cpu())
            all_labels.append(accelerator.gather(labels).detach().cpu())

        all_predictions = torch.cat(all_predictions)[: len(dataset_eval)]
        all_labels = torch.cat(all_labels)[: len(dataset_eval)]

        eval_mse = compute_mse(all_labels, all_predictions)
        eval_r2 = compute_r2(all_labels, all_predictions)
        accelerator.print(f"epoch {epoch}: eval mse={eval_mse:.4f} r2={eval_r2:.4f}")
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "eval/mse": float(eval_mse),
                    "eval/r2": float(eval_r2),
                }
            )

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir / f"epoch_{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    print("Saving model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir / "final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()