from mil.deepmil.train import main as train
from mil.deepmil.writes_results_cross_val import main as writes_validation_results
from mil.deepmil.writes_final_results import main as writes_test_results
from mil.deepmil.utils import print_dict
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
        "--rep", type=int, default=1, help="Number of repetitions for each test sets."
    )
    parser.add_argument(
        "--n_ensemble",
        type=int,
        help="Number of model to ensemble. selected wrt validation results.",
        default=3,
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
    target = pd.read_csv(args.target_path)
    tests = len(set(target["test"].values))

    # choose better default job name
    date_tag = datetime.date.today().strftime("%Y_%m_%d")
    job_post = f"cross_val_{tests}_folds_{args.rep}_repeats_{date_tag}"
    if args.job_name is None:
        args.job_name = job_post
    else:
        args.job_name = args.job_name + "_" + job_post
    print_dict(args.__dict__, name="job args")

    output_dir = os.path.join(args.job_dir, args.job_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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
    for test in range(tests):
        progress_rep = tqdm(
            desc=f"repeat on fold {test+1}",
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
                "--wsi_dir",
                args.wsi_dir,
                "--target_path",
                args.target_path,
                "--target_name",
                args.target_name,
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
            train(known_args=training_args, verbose=args.tqdm)
            progress_rep.update()
        progress_rep.close()
        progress_test.update()
    progress_test.close()
    print("training done!")
    timer.toc()
    # Root of experiment.
    os.chdir(output_dir)
    training_args = ["--n_ensemble", f"{args.n_ensemble}"]
    print(f"write validation results & save n={args.n_ensemble} best models per fold")
    writes_validation_results(training_args)
    print(
        f"assert ensemble performances on test data using n={args.n_ensemble} best models"
    )
    writes_test_results([])
    print(f"test done!")


if __name__ == "__main__":
    main()
