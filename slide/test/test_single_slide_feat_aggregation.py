import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from slide.tile import aggragate_tiles_features
from slide.utils import print_dict
import torch
from slide.slide_encoder.load import encoder_factory
import warnings
from dtime.trackers import timetracker

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
        "--features_path",
        type=str,
        required=True,
        help="Path to the WSI file to process",
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


def main():
    args = parse_arguments()
    print_dict(
        dict=args.__dict__,
        name="args",
    )
    os.makedirs(args.job_dir, exist_ok=True)
    # time tracker
    time = timetracker(verbose=args.clock)
    time.tic()
    print(f"Aggregating features found at: {args.features_path}...")
    print(f"Slide level encoding with {args.encoder}...")
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
    slide_feature_dir = os.path.join(args.job_dir, f"features_{args.encoder}")
    slide_feature_path = aggragate_tiles_features(
        features_path=args.features_path,
        slide_encoder=model,
        device=device,
        dst=slide_feature_dir,
        save_as="h5",
    )
    print(f"Slide level encoding was saved at {slide_feature_path}...")
    time.toc()


if __name__ == "__main__":
    main()
