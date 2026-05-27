from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import yaml


# add choices when needed
def get_arguments(known_args=None, train=True, config=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing the HE tiles to be used as inputs.",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input table split",
    )

    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Path to the target h5 file containing proteomic signatures",
    )

    parser.add_argument(
        "--job_dir", type=str, required=True, help="Directory to store outputs"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel threads for batch processing",
    )

    parser.add_argument(
        "--k_folds",
        type=int,
        default=None,
        help="number of k-folds for cross-validation",
    )

    parser.add_argument(
        "--test_fold", type=int, default=0, help="identifier of the fold used as a test"
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="number of repetitions per fold.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="identifier of the repetition. Used to ID the result.",
    )

    parser.add_argument(
        "--use_val", default=1, help="Use a validation set when training"
    )

    parser.add_argument(
        "--ref_metric",
        type=str,
        default="loss",
        choices=["loss"],
        help="reference metric for validation and storing of the best model.",
    )

    parser.add_argument("--criterion", type=str, default="mse", help="criterion used")

    parser.add_argument("--optimizer", type=str, default="adam")

    # check which model are indeed implemented, delete unnecessary
    parser.add_argument(
        "--encoder",
        type=str,
        default="hoptimus1",
        help="name of the encoder to used.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank to use",
    )
    parser.add_argument(
        "--lora_a",
        type=int,
        default=1,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_target",
        nargs="+",
        default=["qkv"],
        help="LoRA target modules",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536,
        help="Dimension of the embedding space",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="size of the input image",
    )

    parser.add_argument(
        "--lora", type=bool, default=True, help="use lora to fine tune the encoder"
    )

    parser.add_argument(
        "--instance_transf",
        default=False,
        type=bool,
        help="whether to transform the tiles before attention and classification.",
    )

    parser.add_argument(
        "--feature_depth",
        type=int,
        default=512,
        help="Number of features to keep",
    )

    parser.add_argument(
        "--n_layers_classif",
        type=int,
        help="number of the internal layers of the classifier - works with model = mhmc; mlp",
        default=3,
    )

    parser.add_argument(
        "--width_fe",
        type=int,
        help="number of neurons in the internal layers of the classifier",
        default=512,
    )

    parser.add_argument("--dropout", type=float, help="dropout parameter", default=0.4)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch Size = how many WSI in a batch",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for training"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience parameter for early stopping. By default, patience is set to epochs.",
    )

    parser.add_argument("--lr", type=float, help="learning rate", default=0.003)

    # linear is not the best name 
    parser.add_argument("--lr_scheduler", type=str, default="cos", choices=["cos", "linear"])

    parser.add_argument(
        "--patience_lr",
        type=int,
        help="number of epochs for the lr linear decay",
        default=None,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file. If None, use the command line parser",
        default=None,
    )

    parser.add_argument(
        "--write_config", action="store_true", help="writes config in the cwd."
    )

    if not train:  # If test, batch_size=1
        parser.add_argument(
            "--model_path", type=str, required=True, help="Path to the model to load"
        )

    args, _ = parser.parse_known_args(known_args)

    # If there is a config file, we overwrite default args with it
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)

    args.train = train
    args.patience = args.epochs if args.patience is None else args.patience
    args.patience_lr = None if args.lr_scheduler == "cos" else args.patience_lr

    if not args.instance_transf:
        args.feature_depth = args.feature_dim

    # Sgn_metric used to orient the early stopping and writing process.
    if args.ref_metric == "loss":
        args.ref_metric = "mean_val_loss"
        args.sgn_metric = 1
    else:
        args.sgn_metric = -1

    # Writes the config_file
    args_dict = copy.copy(vars(args))
    config_str = yaml.dump(args_dict)
    with open("./config.yaml", "w") as config_file:
        config_file.write(config_str)

    return args
