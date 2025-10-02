from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pkg.predict import main as pred
import pandas as pd
import os
from pkg.utils import print_dict
from dtime.trackers import timetracker


def parse_arguments(raw_args):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_path",
        default=".",
        type=str,
        help="path where the model is stored.",
    )

    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        help="directory where the tile img are stored.",
    )

    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        help="path where the input table is stored.",
    )

    parser.add_argument(
        "--verbose",
        default=0,
        type=int,
        help="verbosity.",
    )

    parser.add_argument(
        "--clock",
        action="store_true",
        default=False,
        help="Display time tracker progress bar",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="prediction_table",
        help="output name",
    )

    return parser.parse_args()


def main(raw_args=None):
    args = parse_arguments(raw_args)
    timer = timetracker(name="tracker", verbose=args.clock)
    print_dict(args.__dict__, name="job args")

    timer.tic()
    results = pred(
        args.model_path, args.img_dir, args.input_path, args.verbose
    )

    out_dir = os.path.join(os.path.dirname(args.model_path), "prediction")
    os.makedirs(out_dir, exist_ok=True)

    result_table = pd.DataFrame(
        data=results["pred"], index=results["img_path"], columns=results["features"]
    )

    result_table.to_csv(
        os.path.join(out_dir, f"{args.name}.csv"),
        index=True,
    )

    print("prediction done!")
    timer.tic()


if __name__ == "__main__":
    main()
