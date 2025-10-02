import os
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
from slide.utils import (
    print_dict,
)
from dtime.trackers import timetracker
from osfile.manager import findFile
import pandas as pd
import numpy as np
from slide.utils import save_h5

warnings.filterwarnings("ignore")


def get_norm(method: str = "identity"):
    if method == "identity":
        return lambda x: np.identity(x)
    if method == "log1p":
        return lambda x: np.log1p(x)
    else:
        KeyError("Unrecognised method!")


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--src",
        type=str,
        help="",
    )

    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="",
    )

    parser.add_argument(
        "--feat",
        type=list,
        nargs="+",
        default=[
            "cells",
            "CD45+",
            "CD68+",
            "CD163+",
            "CD3e+",
            "CD8a+",
            "CD45R0+",
            "CD20+",
            "CD4+",
            "FOXP3+",
            "PDL1+",
            "PD1+",
            "Ki67+",
            "PanCK+",
            "Ecadherin+",
            "CD31+",
            "SMA+",
        ],
        help="",
    )

    parser.add_argument(
        "--norm",
        type=str,
        default="log1p",
        choices=["identity", "log1p"],
        help="normalization to use",
    )

    parser.add_argument(
        "--tqdm", action="store_true", default=False, help="Display tqdm progress bar"
    )

    parser.add_argument(
        "--clock",
        action="store_true",
        default=False,
        help="Display elapsed time of job",
    )

    args = parser.parse_args()
    return args


def main():
    # time tracker
    args = parse_arguments()

    time = timetracker(verbose=args.clock)
    time.tic()

    with open(os.path.join(args.src, "config.yaml"), "r") as f:
        aggr_config = yaml.safe_load(f)
    print_dict(
        dict=aggr_config,
        name="aggregation config file",
    )

    # save master table target .csv
    if args.dst == None:
        target_dir = os.path.join(args.src, "target")
    else:
        target_dir = os.path.join(args.dst)
    os.makedirs(target_dir, exist_ok=True)

    # load proteo tile signatures and concat
    group_dir = os.path.join(args.src, "group")
    group_paths, _ = findFile(group_dir, strings="csv", fileExtensions=True)
    group_df = pd.DataFrame()
    for file in group_paths:
        tmp = pd.read_csv(file)
        loc = tmp.columns.get_loc("coords_path")
        tmp.insert(
            loc=loc,
            column="img_path",
            value=tmp.apply(
                lambda row: row["wsi_name"]
                + f"_{row['x']}_{row['y']}_{row['w']}_{row['h']}.jpeg",
                axis=1,
            ),
        )
        group_df = pd.concat([group_df, tmp], axis=0, ignore_index=True)
    print(f"total nb tiles = {group_df.shape[0]}, across N = {len(group_paths)} files")

    # save master table input .csv (to be splited)
    # stratif / cell count or intensity -> keep cells count (before norm)
    # could stratif / entropy
    img_df = group_df[["orion_id", "wsi_path", "img_path", "cells"]]
    img_df.to_csv(os.path.join(target_dir, f"master_table_input.csv"))

    # normalization
    normalize = get_norm(args.norm)
    group_df_norm = group_df.copy()
    group_df_norm.loc[:, args.feat] = group_df[args.feat].apply(
        lambda x: normalize(x), axis=0
    )

    group_df_norm.to_csv(
        os.path.join(target_dir, f"master_table_target_{args.norm}.csv")
    )

    # Generate target .h5
    target = {
        "signature": group_df_norm[args.feat].to_numpy(),  # N_samples x N_features
        "coords": group_df_norm[["x", "y", "w", "h"]].to_numpy(),  # N_samples x 4
        "img_path": group_df_norm["img_path"].to_numpy(),  # N_samples
        "wsi_path": group_df_norm["wsi_path"].to_numpy(),  # N_samples
    }
    sign_attr = {
        "feat": args.feat,
    }
    coord_attr = {
        "mag": group_df_norm["mag"].unique(),
        "level": group_df_norm["level"].unique(),
    }

    # Save the assets and attributes to an hdf5 file
    patch_file = os.path.join(target_dir, f"target_{args.norm}.h5")

    save_h5(
        patch_file,
        assets=target,
        attributes={"signature": sign_attr, "coords": coord_attr},
        mode="w",
    )

    time.toc()


if __name__ == "__main__":
    main()
