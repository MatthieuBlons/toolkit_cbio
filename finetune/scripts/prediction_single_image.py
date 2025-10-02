from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pkg.predict import predict_one_image, load_model_from_path
import pandas as pd
import os
from pkg.utils import print_dict
from dtime.trackers import timetracker
import torch
import h5py


def parse_arguments(raw_args):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_path",
        default=".",
        type=str,
        help="path where the model is stored.",
    )

    parser.add_argument(
        "--img_path",
        default=None,
        type=str,
        help="directory where the tile img are stored.",
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
        help="Display elapsed time ",
    )

    return parser.parse_args()


def main(raw_args=None):
    args = parse_arguments(raw_args)
    timer = timetracker(name="tracker", verbose=args.clock)

    timer.tic()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device used = {device}")

    model = load_model_from_path(model_path=args.model_path, device=device)
    with h5py.File(model.args.target_path, "r") as f:
        attrs = dict(f["signature"].attrs)
        features = attrs["feat"]

    if args.verbose > 0:
        print_dict(model.args.__dict__, name="training args")
        if model.network.lora:
            model.network.print_lora_summary()
            model.network.print_summary(verbose=args.verbose)

    results = predict_one_image(model, args.img_path)
    prediction = results["pred"]

    out_dir = os.path.join(os.path.dirname(args.model_path), "prediction")
    os.makedirs(out_dir, exist_ok=True)

    result_table = pd.DataFrame(
        data=[prediction], index=[args.img_path], columns=features
    )

    name = os.path.splitext(os.path.basename(args.img_path))[0]

    result_table.to_csv(
        os.path.join(out_dir, f"{name}_prediction.csv"),
        index=True,
    )

    print(f"prediction on {name} done!")
    timer.tic()


if __name__ == "__main__":
    main()
