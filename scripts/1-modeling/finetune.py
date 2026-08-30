"""
Fine-tuning the transformer model on the downstream gene expression prediction task.

This mirrors the Trainer-based flow used in the Kaggle notebook
(gurveersinghvirk/florabert-2): build the model on top of a pretrained
language model, then train a mean-pooling regression head with
`training.make_trainer` + `training.do_training`.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import numpy as np

from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr
from module.florabert.utils import compute_r2, compute_mse

# TPU support (torch_xla). Imported lazily so the script also runs on GPU/CPU.
if os.environ.get("KAGGLE_TPU") or os.environ.get("TPU_NAME"):
    import torch_xla.core.xla_model as xm
else:
    xm = None


DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TRAIN_DATA = "train.tsv"
EVAL_DATA = "eval.tsv"
TEST_DATA = "test.tsv"
DEFAULT_MODEL = "roberta-pred-mean-pool"
# Optional pickled sklearn preprocessor (e.g. Yeoh-Johnson). None = skip;
# the default `transformation="log"` path does not need it.
PREPROCESSOR = None


def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )


def get_device():
    if xm is not None:
        return xm.xla_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        transformation="log",
        hyperparam_search_metrics="mse",
        hyperparam_search_trials=10,
    )

    # Apply model-name-specific defaults unless the user overrode them on the CLI
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

    device = get_device()
    model = model.to(device)

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
    dataset_test = datasets["test"].remove_columns(["sequence"])
    print(f"Loaded training data with {len(dataset_train)} examples")

    data_collator = dataio.load_data_collator("pred")
    training_settings = config.settings["training"]["finetune"]
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs
    print(training_settings)

    model_init = lambda: load_model(args, settings)[2]  # For hyperparameter search
    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_eval,
        args.output_dir,
        hyperparameter_search=args.hyperparameter_search,
        model_init=model_init,
        metrics=args.metrics,
        **training_settings,
    )

    print("Starting training")
    training.do_training(trainer, args, args.output_dir)

    print("Final evaluation")
    metrics = trainer.evaluate(dataset_test)
    print(metrics)

    print("Saving model")
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()