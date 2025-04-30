import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dtime.trackers import timetracker
from slide.pca import fit
from slide.utils import read_h5_features, read_h5_coords, save_h5
from tqdm import tqdm

# For the sklearn warnings
import warnings


warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--feat_dir", type=str, required=True, help="Directory to the .h5 feat files"
    )

    parser.add_argument(
        "--encoder_name",
        type=str,
        required=True,
        help="name of the encoder used to extract features",
    )

    parser.add_argument(
        "--msg", action="store_true", default=False, help="Write summary message"
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
    timer = timetracker(name="tracker", verbose=args.clock)
    assert os.path.isdir(
        os.path.join(args.feat_dir, f"features_{args.encoder_name}")
    ), f"Make sure features were extracted with encoder: {args.encoder_name}"
    output_dir = os.path.join(args.feat_dir, f"features_pca_{args.encoder_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    timer.tic()
    print("Fit PCA...")
    files, ipca = fit(os.path.join(args.feat_dir, f"features_{args.encoder_name}"))

    # prog bar
    progress = tqdm(
        files,
        desc=f"Transf PCA",
        total=len(files),
        unit="wsi",
        initial=0,
        position=0,
        leave=True,
        disable=not args.tqdm,
    )

    for file in progress:
        feat_attrs, feat = read_h5_features(file)
        name = feat_attrs["name"]
        coord_attrs, coord = read_h5_coords(file)
        transf = ipca.transform(feat)
        transf_path = os.path.join(output_dir, f"{name}.h5")
        assets = {"features": transf, "coords": coord}
        attributes = {
            "features": {
                "encoder": args.encoder_name,
                "name": name,
                "dst": output_dir,
            },
            "coords": coord_attrs,
        }
        save_h5(
            transf_path,
            assets=assets,
            attributes=attributes,
            mode="w",
        )
        progress.set_postfix_str(f"wsi: {name}", refresh=True)
        progress.update()

    if args.msg:
        msg = " ----------------  RESULTS -------------------- \n"
        s = 0
        for i, o in enumerate(ipca.explained_variance_ratio_, 1):
            s += o
            msg += "Dimensions until {} explains {}% of the variance \n".format(
                i, s * 100
            )
        msg += "----------------------------------------------"
        with open("./pca_results.txt", "w") as f:
            f.write(msg)
    print("Done!")
    timer.toc()


if __name__ == "__main__":
    main()
