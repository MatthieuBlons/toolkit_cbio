import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os


def test_stratif(table, equ_vars, group, k):
    """
    divide into k-fold the dataset (creation of test_set)
    according to all the variables in equ_vars.
    """
    table = pd.read_csv(table)
    if type(equ_vars) != list:
        equ_vars = [equ_vars]
    equ_vars = equ_vars
    new_var = table.apply(make_new_var(equ_vars), axis=1)
    table["stratif"] = new_var

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True)
    y = table["stratif"].values
    groups = table[group].values
    X = list(range(len(y)))

    test_vec = np.zeros(len(y))
    for o, (train, test) in enumerate(sgkf.split(X, y, groups)):
        for i in test:
            test_vec[i] = o
    table["test"] = test_vec
    return table


def make_new_var(list_vars):
    def get_func(x):
        new = ""
        for v in list_vars:
            if v is not None:
                new += str(x[v]) + "_"
        return new

    return get_func


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--job_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--table_path", type=str, required=True)
    parser.add_argument("--group_name", type=str, required=True)
    parser.add_argument(
        "--equ_vars",
        type=str,
        default=None,
        nargs="+",
        help="variables to keep in stratif vars (will be used for balancing the dataset)",
    )
    parser.add_argument("-k", type=int, default=5, help="number of folds")
    return parser.parse_args()


def main():
    args = parse_arguments()
    equ_vars = args.equ_vars
    table = test_stratif(args.table_path, equ_vars, args.group_name, args.k)

    if not args.job_dir:
        output_path = args.table_path
    else:
        table_name = os.path.basename(args.table_path)
        output_path = os.path.join(args.job_dir, table_name)

    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
