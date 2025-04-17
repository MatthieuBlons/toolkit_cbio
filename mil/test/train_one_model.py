from mil.deepmil.train import main as train
from mil.deepmil.utils import print_dict
import os
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
from dtime.trackers import timetracker
from torch import cuda
import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Enable GPU acceleartion"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Device ID to use for encoding tiles"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for parallelization",
    )

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
        "--job_name",
        type=str,
        help="Name of the experiment, current date by default",
        default=None,
    )
    parser.add_argument(
        "--job_dir", type=str, required=True, help="Directory to store outputs"
    )
    parser.add_argument(
        "--test_fold", type=int, default=0, help="Idx of test fold to use"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config_default.yaml",
        help="Config file for lazy args passing",
    )
    parser.add_argument(
        "--tqdm", action="store_true", default=False, help="Display tqdm progress bar"
    )
    parser.add_argument(
        "--clock",
        action="store_true",
        default=False,
        help="Display time tracker progress bar",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.gpu:
        if cuda.is_available():
            device = f"cuda:{args.device}"
        else:
            print(f"cuda is available: {cuda.is_available()}, device set to: cpu")
            device = "cpu"
    else:
        device = "cpu"

    timer = timetracker(name="tracker", verbose=args.clock)

    try:
        with open(args.config, "r") as f:
            config_mil = yaml.safe_load(f)
            args.__dict__.update(config_mil)
    except FileNotFoundError:
        print("could not find config file, will use deflaut mil param")

    if "n_tiles" in args.__dict__.values():
        n_tiles = args["n_tiles"]
        print(f"will use n = {n_tiles} tiles per wsi for training & validation")

    # choose better default job name
    date_tag = datetime.date.today().strftime("%Y_%m_%d")
    job_post = f"single_model_{date_tag}"
    if args.job_name is None:
        args.job_name = job_post
    else:
        args.job_name = args.job_name + "_" + job_post
    print_dict(args.__dict__, name="job args")

    output_dir = os.path.join(args.job_dir, args.job_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    os.chdir(output_dir)
    training_args = [
        "--device",
        device,
        "--num_workers",
        f"{args.num_workers}",
        "--wsi_dir",
        args.wsi_dir,
        "--target_path",
        args.target_path,
        "--target_name",
        args.target_name,
        "--job_dir",
        output_dir,
        "--test_fold",
        f"{args.test_fold}",
        "--config",
        args.config,
    ]
    timer.tic()
    train(known_args=training_args, verbose=args.tqdm)
    print(f"training done!")
    timer.tic()


if __name__ == "__main__":
    main()
