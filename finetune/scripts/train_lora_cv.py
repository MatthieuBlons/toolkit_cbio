from pkg.train import main as train
from pkg.utils import print_dict
import pandas as pd
import os
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
from dtime.trackers import timetracker
from tqdm import tqdm
from torch import cuda


# For the sklearn warnings
import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Enable GPU acceleartion"
    )
    parser.add_argument("--device", type=int, default=0, help="GPU id")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers for parallelization",
    )
    parser.add_argument(
        "--use_fold",
        type=int,
        default=0,
        help="a single cv folds to use",
    )

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
        "--job_name",
        type=str,
        help="Name of the experiment, current date by default",
        default="lora",
    )
    parser.add_argument(
        "--job_dir", type=str, required=True, help="Directory to store outputs"
    )
    parser.add_argument(
        "--rep", type=int, default=1, help="Number of repetitions for each test sets."
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
        print("could not find config file, will use deflaut param")

    input_imgs = pd.read_csv(args.input_path)

    if args.use_fold is None:
        tests = len(set(input_imgs["test"].values))
        folds = [t for t in range(tests)]
    else:
        tests = 1
        folds = [args.use_fold]

    # choose better default job name
    date_tag = datetime.date.today().strftime("%Y_%m_%d")
    job_post = f"cv_{tests}_folds_{args.rep}_repeats_{date_tag}"
    if args.job_name is None:
        args.job_name = job_post
    else:
        args.job_name = args.job_name + "_" + job_post

    output_dir = os.path.join(args.job_dir, args.job_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print_dict(args.__dict__, name="job args")

    progress_test = tqdm(
        desc=f"cross val",
        total=tests,
        unit="test fold",
        initial=0,
        position=0,
        leave=True,
        disable=not args.tqdm,
    )
    timer.tic()
    for test in folds:
        progress_rep = tqdm(
            desc=f"repeat on fold {test}",
            total=args.rep,
            unit="repeat",
            initial=0,
            position=1,
            leave=False,
            disable=not args.tqdm,
        )
        for rep in range(args.rep):
            wd = os.path.join(output_dir, f"test_{test}", f"rep_{rep}")
            os.makedirs(wd, exist_ok=True)
            os.chdir(wd)
            training_args = [
                "--device",
                device,
                "--num_workers",
                f"{args.num_workers}",
                "--img_dir",
                args.img_dir,
                "--input_path",
                args.input_path,
                "--target_path",
                args.target_path,
                "--job_dir",
                wd,
                "--reps",
                f"{args.rep}",
                "--repeat",
                f"{rep}",
                "--k_folds",
                f"{tests}",
                "--test_fold",
                f"{test}",
                "--config",
                args.config,
            ]
            stop_epoch = train(known_args=training_args, verbose=args.tqdm)
            progress_rep.set_postfix_str(f"STOP EPOCH={stop_epoch}/{args.epochs}")
            progress_rep.update()
        progress_rep.close()
        progress_test.update()
    progress_test.close()
    print("training done!")
    timer.toc()


if __name__ == "__main__":
    main()
