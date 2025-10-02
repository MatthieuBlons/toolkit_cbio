from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pkg.test import main as test
import pandas as pd
import os
from pkg.utils import print_dict
from dtime.trackers import timetracker


def fill_table(table, preds):
    """fill_table"""

    return table


def parse_arguments(raw_args):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_path",
        default=".",
        type=str,
        help="path where the model is stored.",
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

    return parser.parse_args()


def main(raw_args=None):
    args = parse_arguments(raw_args)
    timer = timetracker(name="tracker", verbose=args.clock)
    print_dict(args.__dict__, name="job args")

    timer.tic()
    results = test(args.model_path, args.input_path, args.verbose)

    input_table = pd.read_csv(args.input_path, index_col=0)

    result_table = pd.DataFrame(
        data=results["pred"], index=[results["img_path"]], columns=results["features"]
    )

    final_tabel = pd.merge(
        left=input_table,
        right=result_table,
        right_index=True,
        left_on="img_path",
        how="left",
    )

    result_table.to_csv(
        os.path.join(os.path.dirname(args.model_path), "results_table.csv"), index=True
    )

    final_tabel.to_csv(
        os.path.join(os.path.dirname(args.model_path), "final_tabel.csv"), index=False
    )

    print("test done!")
    timer.tic()


if __name__ == "__main__":
    main()
