from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import copy
import yaml


# add choices when needed
def get_arguments(known_args=None, train=True, config=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Path to the WSI feature file to use as inputs",
    )

    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Path to the target table",
    )

    parser.add_argument(
        "--target_name",
        type=str,
        required=True,
        help="Name of the target variable as referenced in target_path.",
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
        "--k_folds", type=int, default=None, help="number of k-folds for cross-validation"
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
        "--n_tiles",
        type=int,
        default=300,
        help="Number of tiles per WSI. If 0, the whole slide is processed.",
    )

    parser.add_argument(
        "--sampler",
        type=str,
        help="Type of tile sampler. dispo : all | random | random_strict",
        default="random",
    )

    parser.add_argument(
        "--sample_wr_whole_label",
        type=int,
        help="if 1, each class is equally probable. Else, sampling is done s.t. the conditional wr to the target are equals.",
        default=0,
    )

    parser.add_argument(
        "--no_strat_sampling",
        default=0,
        type=int,
        help="if = 1, do not use strategic sampling - even to balance the dataset -",
    )

    parser.add_argument(
        "--use_val", default=1, help="Use a validation set when training"
    )
    parser.add_argument(
        "--val_sampler", default="all", help="tile sampler to use for validation"
    )

    parser.add_argument(
        "-k", type=int, default=5, help="k parameter to select topk and lowk tiles"
    )

    parser.add_argument(
        "--ref_metric",
        type=str,
        default="loss",
        help="reference metric for validation, early stoping, storing of the best model.",
    )

    parser.add_argument("--criterion", type=str, help="criterion used", default="nll")

    parser.add_argument("--optimizer", type=str, default="adam")

    # check which model are indeed implemented, delete unnecessary
    parser.add_argument(
        "--model",
        type=str,
        default="mhmc_layers",
        choices=[
            "generalmil",
            "multiheadmil",
            "mhmc_layers",
            "conan",
            "1s",
            "sa",
            "transformermil",
        ],
        help="name of the model used. Avail : mhmc_layers | 1s | transformermil | sa | conan ",
    )

    parser.add_argument(
        "--instance_transf",
        default=0,
        type=int,
        help="either 1 or 0, wether to transform the tiles before classification and attention",
    )

    parser.add_argument(
        "--atn_dim",
        type=int,
        help="intermediate projection dimension during attention mechanism. must be divisible by num_heads.",
        default=256,
    )
    parser.add_argument(
        "--num_heads", help="number of attention_head", default=1, type=int
    )

    parser.add_argument(
        "--pooling_fct",
        type=str,
        default="ilse",
        help="pooling function used. max, mean, ilse, conan possible",
    )

    parser.add_argument(
        "--feature_depth", type=int, default=512, help="Number of features to keep"
    )

    parser.add_argument(
        "--n_layers_classif",
        type=int,
        help="number of the internal layers of the classifier - works with model_name = mhmc_layers",
        default=3,
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

    parser.add_argument("--lr_scheduler", type=str, default="cos")

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

    if not train:  # If test, n_tiles = 0 (all tiles considered) and batch_size=1
        parser.add_argument(
            "--model_path", type=str, required=True, help="Path to the model to load"
        )

    args, _ = parser.parse_known_args(known_args)

    # If there is a config file, we overwrite default args with it
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)

    # continue to ppolulate args
    target_table = pd.read_csv(args.target_path)
    args.num_class = len(set(target_table[args.target_name]))
    args.train = train
    args.patience = args.epochs if args.patience is None else args.patience
    args.patience_lr = None if args.lr_scheduler == "cos" else args.patience_lr

    # Set constant size flag
    if args.n_tiles == 0 or args.sampler == "random_strict":
        args.constant_size = False
    else:
        args.constant_size = True

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
