import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from slide.tile import SlidePatcher, TileEncoder
from slide.utils import get_slide_reader, print_dict
import torch
from slide.tile_encoder.load import encoder_factory
import yaml
import warnings
from dtime.trackers import timetracker
import copy

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
        default=8,
        help="number of workers for parallelization",
    )

    parser.add_argument(
        "--slide_path",
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
        choices=[
            "conch_v1",
            "uni_v1",
            "uni_v2",
            "ctranspath",
            "phikon",
            "resnet50",
            "gigapath",
            "virchow",
            "virchow2",
            "hoptimus0",
            "hoptimus1",
            "phikon_v2",
            "conch_v15",
            "musk",
        ],
        default="uni_v2",
        help="Tile encoder to use for feature extraction",
    )
    parser.add_argument(
        "--batch_max",
        type=int,
        default=32,
        help="Maximun batch lenght for feature extration",
    )
    parser.add_argument(
        "--target_mag",
        type=int,
        default=20,
        help="Target magnification at which patches/features are extracted",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of patches in pixels to be extracted",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Patch overlap in pixels",
    )
    parser.add_argument(
        "--seg_down",
        type=int,
        default=32,
        help="Downsampling factor to use for tissue segmentation",
    )
    parser.add_argument(
        "--seg_tolerence",
        type=float,
        default=0.5,
        help="Min tissue proportion wrt background in valid patches",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config_default.yaml",
        help="Config file for lazy args passing",
    )

    parser.add_argument(
        "--tqdm", action="store_true", default=False, help="display tqdm progress bar"
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
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)
    # time tracker
    time = timetracker(verbose=args.clock)
    time.tic()
    print(f"Processing slide at: {args.slide_path}...")
    reader = get_slide_reader(args.slide_path)
    slide = reader(args.slide_path)
    # Extract patches from the wsi
    print("Extracting valid tissue patches...")
    out_dir = os.path.join(
        args.job_dir,
        f"tile_feat_{args.target_mag}x_{args.patch_size}px_{args.overlap}px_overlap",
    )
    os.makedirs(out_dir, exist_ok=True)
    print_dict(
        dict=args.__dict__,
        name="args",
    )
    patcher = SlidePatcher(
        slide,
        mag_target=args.target_mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        mask_downsample=args.seg_down,
        mask_tolerance=args.seg_tolerence,
        xywh_only=False,
        dst=out_dir,
    )
    # save tissue segmentation
    seg_dir = os.path.join(out_dir, "segmentation")
    seg_path = patcher.visualize_tissue_seg(
        size=(1024, 1024), save_seg=seg_dir, show=True
    )
    print(f"You can visualize tissue segemtation in: {seg_path}")
    # save visualization cut
    visu_dir = os.path.join(out_dir, "visualization")
    cut_path = patcher.visualize_cut(size=(1024, 1024), save_cut=visu_dir, show=True)
    print(f"You can visualize patch extraction in: {cut_path}")
    # Encode patches with encoder
    print(f"Encoding valid tissue patches with {args.encoder}...")
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
    encoder = TileEncoder(
        slide,
        tile_encoder=model,
        coords_path=os.path.join(out_dir, "patches", f"{slide.name}_patches.h5"),
        device=device,
        num_workers=args.num_workers,
        batch_max=args.batch_max,
        feat_only=False,
        dst=out_dir,
        verbose=args.tqdm,
    )
    # save features umap visualization
    umap_dir = os.path.join(out_dir, f"cluster_{args.encoder}")
    umap_path = encoder.visualize_feat(
        pcs=50, neighbors=20, resolution=0.2, save_cluster=umap_dir
    )
    print(f"You can visualize a umap of features in: {umap_path}")
    print(f"Feature extraction completed. Results saved to {out_dir}")
    time.toc()
    # Writes the config.yaml file in output directory
    config_str = yaml.dump(copy.copy(vars(args)))
    os.chdir(os.path.dirname(encoder.feat_path))
    with open("config.yaml", "w") as config_file:
        config_file.write(config_str)


if __name__ == "__main__":
    main()
