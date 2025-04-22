import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from slide.tile import SlidePatcher, TileEncoder
from slide.utils import get_slide_reader, print_dict
import torch
from slide.load import encoder_factory
import yaml
from tqdm import tqdm
from glob import glob
import copy
from dtime.trackers import timetracker

import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Enable GPU acceleartion"
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
        "--wsi_dir",
        type=str,
        required=True,
        help="Path to the WSI file to process",
    )

    parser.add_argument(
        "--ext",
        type=str,
        choices=["ndpi", "svs", "tif"],
        default="ndpi",
        help="WSI file extension",
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
        "--tqdm", action="store_true", default=False, help="Display tqdm progress bar"
    )

    parser.add_argument(
        "--clock",
        action="store_true",
        default=False,
        help="Display elapsed time of job",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    # If there is a config file we immediately, we populate args with it (keeping the default arguments if not in config)
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)

    output_dir = os.path.join(
        args.job_dir,
        f"tile_feat_{args.target_mag}x_{args.patch_size}px_{args.overlap}px_overlap",
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # make function to print dic -h
    print_dict(
        dict=args.__dict__,
        name="args",
    )
    # get slide
    all_wsi_wth_ext = glob(os.path.join(args.wsi_dir, f"*.{args.ext}*"))
    assert (
        len(all_wsi_wth_ext) > 0
    ), f"no wsi with extension {args.ext} in src {args.wsi_dir}"
    print(f"num wsi = {len(all_wsi_wth_ext)}")

    model = encoder_factory(args.encoder)
    if args.gpu:
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    model.to(device)
    used_memory = torch.cuda.memory_allocated()
    print(f"Memory allocated for {args.encoder}: {used_memory / (1024 ** 2):.2f} MB")
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Free memory: {free_mem / (1024 ** 2):.2f} MB")
    print(f"Total memory: {total_mem / (1024 ** 2):.2f} MB")

    # enc process already? make a fucntion
    features_dir = os.path.join(output_dir, f"features_{args.encoder}")
    processed_already = glob(os.path.join(features_dir, "*.h5*"))
    wsi_to_exclude = [
        os.path.basename(os.path.splitext(p)[0])[:-6] for p in processed_already
    ]
    print(
        f"num wsi (total) = {len(wsi_to_exclude)} have aleardy been processed wth {args.encoder}... "
    )

    wsi_to_process = [
        s for s in all_wsi_wth_ext if not any([True for t in wsi_to_exclude if t in s])
    ]

    # time tracker
    time = timetracker(verbose=args.clock)
    # prog bar
    progress = tqdm(
        wsi_to_process,
        desc=f"batch enc wth {args.encoder}",
        total=len(all_wsi_wth_ext),
        unit="wsi",
        initial=0,
        position=0,
        leave=True,
        disable=not args.tqdm,
    )
    time.tic()
    for wsi_path in progress:

        reader = get_slide_reader(wsi_path)
        slide = reader(wsi_path)
        name = slide.name

        patcher = SlidePatcher(
            slide,
            mag_target=args.target_mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            mask_downsample=args.seg_down,
            mask_tolerance=args.seg_tolerence,
            xywh_only=False,
            dst=output_dir,
        )

        # save tissue segmentation
        seg_dir = os.path.join(output_dir, "segmentation")
        _ = patcher.visualize_tissue_seg(
            size=(1024, 1024), save_seg=seg_dir, show=False
        )

        # save visualization cut
        visu_dir = os.path.join(output_dir, "visualization")
        _ = patcher.visualize_cut(size=(1024, 1024), save_cut=visu_dir, show=False)

        # encode tiles
        encoder = TileEncoder(
            slide,
            tile_encoder=model,
            coords_path=os.path.join(output_dir, "patches", f"{name}_patches.h5"),
            device=device,
            num_workers=args.num_workers,
            batch_max=args.batch_max,
            feat_only=False,
            dst=output_dir,
            verbose=args.tqdm,
        )

        # save features umap visualization
        umap_dir = os.path.join(output_dir, f"cluster_{args.encoder}")
        _ = encoder.visualize_feat(
            pcs=50, neighbors=20, resolution=0.2, save_cluster=umap_dir
        )

        progress.set_postfix_str(f"wsi: {name}", refresh=True)
        progress.update()

    progress.clear()
    # Writes the config.yaml file in output directory
    config_str = yaml.dump(copy.copy(vars(args)))
    os.chdir(output_dir)
    with open("config.yaml", "w") as config_file:
        config_file.write(config_str)

    print(f"Feature extraction done! Results saved to {output_dir}")
    time.toc()


if __name__ == "__main__":
    main()
