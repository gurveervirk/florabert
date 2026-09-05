"""
Pretraining on masked language model task.
"""
import sys
sys.path.append('/kaggle/working/florabert')
import torch
import os
from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr


DATA_DIR = config.data_final / "transformer" / "seq"
DEFAULT_MODEL = "roberta-lm"


def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data="all_seqs_train.txt",
        test_data="all_seqs_test.txt",
        output_dir=config.model_output_dir(DEFAULT_MODEL, "language-model"),
        model_name=DEFAULT_MODEL,
        pretrained_model=None,
    )
    OUTPUT_DIR = config.model_output_dir(args.model_name, "language-model")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Apply model-name-specific defaults unless the user overrode them on the CLI
    if "--output-dir" not in sys.argv:
        args.output_dir = OUTPUT_DIR
    # Pretrain from scratch unless a pretrained model/checkpoint is explicitly
    # provided (resume is handled via --warmstart + --pretrained-model).
    if "--pretrained-model" not in sys.argv:
        args.pretrained_model = None
#     args.warmstart = True
    print(args)

    settings = utils.get_model_settings(config.settings, args, args.model_name)


    config_obj, tokenizer, model = tr.load_model(
        args.model_name,
        config.tokenizer_dir_for_model(args.model_name),
        pretrained_model=args.pretrained_model,
        **settings,
    )

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")

    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        test_data=args.test_data,
        file_type="text",
        seq_key="text",
    )
    dataset_train = datasets["train"]
    dataset_test = datasets["test"]
    print(f"Loaded training data with {len(dataset_train):,} examples")
    data_collator = dataio.load_data_collator(
        "language-model",
        tokenizer=tokenizer,
    )

    training_settings = config.settings["training"]["pretrain"]
    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_test,
        args.output_dir,
        **training_settings,
    )

    print(f"Starting training on {torch.cuda.device_count()} GPUs" if "COLAB_TPU_ADDR" not in os.environ else "Starting TPU training")
    training.do_training(trainer, args, args.output_dir)

    print("Saving model")

    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
