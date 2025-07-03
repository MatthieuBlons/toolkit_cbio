import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from slide.tile import aggragate_tiles_features
from slide.utils import print_dict
import torch
from slide.slide_encoder.load import encoder_factory
import warnings
from dtime.trackers import timetracker
from glob import glob
from tqdm import tqdm

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
        "--features_dir",
        type=str,
        required=True,
        help="Directory to the .h5 feat files",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default="h5",
        help="Tile features file extension",
    )

    parser.add_argument(
        "--job_dir", type=str, required=True, help="Directory to store outputs"
    )

    parser.add_argument(
        "--encoder",
        type=str,
        choices=["threads", "titan", "prism", "chief", "gigapath", "madeleine"],
        default="titan",
        help="Tile encoder to use for feature extraction",
    )

    parser.add_argument(
        "--tqdm",
        action="store_true",
        default=False,
        help="Display progress bar",
    )

    parser.add_argument(
        "--clock",
        action="store_true",
        default=False,
        help="Display elapsed time of job",
    )

    parser.add_argument(
        "--model_summary",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Display model summary",
    )
    return parser.parse_args()


# could use a dataloader
def main():
    args = parse_arguments()
    print_dict(
        dict=args.__dict__,
        name="args",
    )
    os.makedirs(args.job_dir, exist_ok=True)

    # get feat files to process
    all_files_wth_ext = glob(os.path.join(args.features_dir, f"*.{args.ext}*"))
    assert (
        len(all_files_wth_ext) > 0
    ), f"no tile feature files with extension {args.ext} in src {args.features_dir}"
    print(
        f"N = {len(all_files_wth_ext)} features.{args.ext} found in {args.features_dir}..."
    )
    # processed already?
    slide_feature_dir = os.path.join(args.job_dir, f"features_{args.encoder}")
    processed_already = glob(os.path.join(slide_feature_dir, "*.h5*"))
    files_to_exclude = [
        os.path.basename(os.path.splitext(p)[0]) for p in processed_already
    ]
    print(
        f"N = {len(files_to_exclude)} feature files have aleardy been processed wth {args.encoder}... "
    )
    file_to_process = [
        file
        for file in all_files_wth_ext
        if not any([True for t in files_to_exclude if t in file])
    ]
    print(f"N (total) = {len(file_to_process)} feature files will be processed... ")

    model = encoder_factory(args.encoder)
    if args.gpu:
        device = f"cuda:{args.device}"
    else:
        device = "cpu"
    model.to(device)
    model.print_summary(verbose=args.model_summary)
    used_memory = torch.cuda.memory_allocated()
    print(f"Memory allocated for {args.encoder}: {used_memory / (1024 ** 2):.2f} MB")
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Free memory: {free_mem / (1024 ** 2):.2f} MB")
    print(f"Total memory: {total_mem / (1024 ** 2):.2f} MB")
    # time tracker
    time = timetracker(verbose=args.clock)
    # prog bar
    progress = tqdm(
        file_to_process,
        desc=f"slide enc wth {args.encoder}",
        total=len(file_to_process),
        unit="wsi",
        initial=0,
        position=0,
        leave=True,
        disable=not args.tqdm,
    )
    time.tic()
    for feat in progress:
        name = os.path.basename(feat)
        _ = aggragate_tiles_features(
            features_path=feat,
            slide_encoder=model,
            device=device,
            dst=slide_feature_dir,
            save_as="h5",
        )
        progress.set_postfix_str(f"feat: {name}", refresh=True)
        progress.update()
    print(f"Tile feature aggregation done! Results saved to {slide_feature_dir}")
    time.toc()


if __name__ == "__main__":
    main()
