import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from slide.tile import SlidePatcher, TileEncoder, read_h5_coords
from slide.utils import get_slide_reader, print_dict
import torch
from slide.tile_encoder.load import encoder_factory
import yaml
from tqdm import tqdm
from glob import glob
from dtime.trackers import timetracker
import pandas as pd
import cv2
import warnings
from mil.deepmil.utils import is_in_args

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="./config_default.yaml",
        help="Config file for lazy args passing",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Enable GPU acceleartion"
    )
    parser.add_argument("--device", type=int, default=None, help="GPU id")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers for parallelization",
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Path to the WSI file to process",
    )
    parser.add_argument(
        "--wsi_list", type=list, nargs="+", help="List of WSI to process", default=None
    )
    parser.add_argument(
        "--ext",
        type=str,
        choices=["ndpi", "svs", "tif"],
        default="tif",
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
            "prov_gigapath",
            "virchow",
            "virchow2",
            "hoptimus0",
            "hoptimus1",
            "phikon_v2",
            "conch_v15",
            "musk",
        ],
        default="hoptimus1",
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
        "--mask_down",
        type=int,
        default=32,
        help="Downsampling factor to use for tissue segmentation",
    )
    parser.add_argument(
        "--mask_tolerance",
        type=float,
        default=0.5,
        help="Min tissue proportion wrt background in valid patches",
    )
    parser.add_argument(
        "--custom_coords",
        type=str,
        default=None,
        help="path to the custom coords folder",
    )
    parser.add_argument(
        "--save_img", type=bool, default=False, help="Save patches as img.jpeg"
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
    parser.add_argument(
        "--model_summary",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Display model summary",
    )

    args = parser.parse_args()
    # If there is a config file, we populate args with it (keeping the default arguments if not in config)
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)

    return args


def main():
    args = parse_arguments()
    # output directory
    output_dir = os.path.join(
        args.job_dir,
        f"tile_feat_{args.target_mag}x_{args.patch_size}px_{args.overlap}px_overlap",
    )
    os.makedirs(output_dir, exist_ok=True)

    print_dict(
        dict=args.__dict__,
        name="args",
    )

    if args.gpu:
        device = "cuda"
        if args.device:
            device += f":{args.device}"
    else:
        device = "cpu"
    embeddings = is_in_args(args, "embeddings", False)

    model = encoder_factory(args.encoder, embeddings=embeddings, device=device)
    model.to(device)
    model.eval()
    model.print_summary(verbose=args.model_summary)
    if args.device:
        free_mem, total_mem = torch.cuda.mem_get_info(device=device)
        print(
            f"Free memory: {free_mem / (1024 ** 2):.2f} / {total_mem / (1024 ** 2):.2f} MB"
        )

    # get slides to process
    all_wsi_wth_ext = glob(
        os.path.join(args.wsi_dir, f"**/*.{args.ext}*"), recursive=True
    )
    print(f"N = {len(all_wsi_wth_ext)} wsi.{args.ext} found in {args.wsi_dir}")
    if args.wsi_list:
        print(f"A list of wsi was provided: {args.wsi_list}")
        wsi_table = pd.read_csv(args.wsi_list)
        wsi_in_list = wsi_table["wsi_path"].to_list()
        wsi_to_process = [wsi for wsi in all_wsi_wth_ext if wsi in wsi_in_list]
    else:
        wsi_to_process = all_wsi_wth_ext
    assert (
        len(wsi_to_process) > 0
    ), f"No wsi with extension {args.ext} in src {args.wsi_dir}"

    # processed already?
    features_dir = os.path.join(output_dir, f"features_{model.enc_name}")
    processed_already = glob(os.path.join(features_dir, "*.h5*"))
    # intersect basename(wsi_to_process) and processed_already
    wsi_to_exclude = [
        os.path.basename(os.path.splitext(p)[0]) for p in processed_already
    ]
    print(
        f"N = {len(wsi_to_exclude)} wsi have aleardy been processed wth {model.enc_name}... "
    )
    wsi_to_process = [
        wsi
        for wsi in wsi_to_process
        if not any([True for t in wsi_to_exclude if t in wsi])
    ]
    print(f"N (total) = {len(wsi_to_process)} wsi.{args.ext} will be processed")
    
    # time tracker
    time = timetracker(verbose=args.clock)
    # prog bar
    progress = tqdm(
        wsi_to_process,
        desc=f"Tile enc wth {args.encoder}",
        total=len(wsi_to_process),
        unit="wsi",
        initial=0,
        position=0,
        leave=True,
        disable=not args.tqdm,
    )
    time.tic()
    for wsi_path in progress:
        reader = get_slide_reader(wsi_path)
        try:
            slide = reader(wsi_path)
        except Exception as e:
            print(f"Error while trying to open wsi: {e}")
            print(f"{wsi_path} will ignored")
            continue

        if args.custom_coords is not None:
            coords_path = os.path.join(args.custom_coords, f"{slide.name}_patches.h5")
            attrs, coords = read_h5_coords(coords_path)
            args.target_mag = attrs["target_magnification"]
            args.patch_size = attrs["target_patch_size"]
            args.overlap = attrs["target_overlap"]
            args.mask_tolerance = attrs["tissu_thr"]
        else:
            coords = None

        patcher = SlidePatcher(
            slide,
            pixel_size_0=slide.mpp,
            pixel_size_target=None,
            mag_0=slide.magnification,
            mag_target=args.target_mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            mask_downsample=args.mask_down,
            mask_tolerance=args.mask_tolerance,
            custom_xywh=coords,
            xywh_only=False,
            pil=False,
            dst=output_dir,
        )

        # save images
        if args.save_img:
            he_dir = os.path.join(output_dir, "tile")
            os.makedirs(he_dir, exist_ok=True)
            for tile, (x, y, w, h) in patcher:
                image_name = slide.name + f"_{x}_{y}_{w}_{h}.jpeg"
                image_path = os.path.join(he_dir, image_name)
                cv2.imwrite(image_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        # save tissue segmentation
        seg_dir = os.path.join(output_dir, "segmentation")
        _ = patcher.visualize_tissue_seg(
            size=(1024, 1024), save_seg=seg_dir, show=False
        )

        # save visualization cut
        visu_dir = os.path.join(output_dir, "visualization")
        _ = patcher.visualize_cut(size=(1024, 1024), save_cut=visu_dir, show=False)

        # encode tiles and save to .h5
        # get patch_path from patcher
        encoder = TileEncoder(
            slide,
            tile_encoder=model,
            coords_path=patcher.patch_path,
            device=device,
            num_workers=args.num_workers,
            batch_max=args.batch_max,
            feat_only=False,
            dst=output_dir,
            verbose=args.tqdm,
        )

        # save features umap
        try:
            umap_dir = os.path.join(output_dir, f"cluster_{model.enc_name}")
            _ = encoder.visualize_feat(
                pcs=0, neighbors=50, resolution=0.3, save_cluster=umap_dir
            )
        except Exception as e:
            print(f"Error while generating umap: {e}")
            continue

        progress.set_postfix_str(f"wsi: {slide.name}", refresh=True)
        progress.update()
    progress.clear()

    print(f"Feature extraction done! Results saved to {output_dir}")
    time.toc()


if __name__ == "__main__":
    main()
