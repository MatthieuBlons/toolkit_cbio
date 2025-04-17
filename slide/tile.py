import numpy as np
from slide.utils import get_size_to, get_x_y_to, make_auto_mask
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional
from slide.utils import grid_blob, mask_percentage, save_h5, read_h5_coords
from slide.draw import visualise_tile_feat
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import h5py
import warnings


class SlidePatcher:
    def __init__(
        self,
        slide,
        pixel_size_0: Optional[float] = None,
        pixel_size_target: Optional[float] = None,
        mag_0: Optional[int] = None,
        mag_target: Optional[int] = 20,
        patch_size: Optional[int] = 224,
        overlap: Optional[int] = 0,
        mask_downsample: Optional[int] = None,
        mask_tolerance: Optional[int] = None,
        custom_xywh: Optional[str] = None,
        xywh_only: Optional[bool] = True,
        pil: Optional[bool] = False,
        dst: Optional[str] = None,
        save_as: Optional[str] = "h5",
        lazy: Optional[bool] = True,
    ):
        self.slide = slide
        self.width, self.height = slide.dimensions
        self.patch_size_target = patch_size
        self.mag_0 = mag_0
        if not mag_0:
            self.mag_0 = slide.magnification
        self.mag_target = mag_target
        downsample = self.mag_0 / self.mag_target
        self.level_target = self.slide.get_best_level_for_downsample(downsample)
        self.downsample = self.get_true_downsample(self.level_target)
        self.pixel_size_0 = pixel_size_0
        if not pixel_size_0:
            self.pixel_size_0 = slide.mpp
        self.pixel_size_target = pixel_size_target
        if not pixel_size_target:
            self.pixel_size_target = self.pixel_size_0 * self.downsample
        self.patch_size_0 = round(self.patch_size_target * self.downsample)
        self.overlap_target = overlap
        self.overlap_0 = round(self.overlap_target * self.downsample)
        self.mask_downsample = mask_downsample
        self.mask_tolerance = mask_tolerance
        self.custom_xywh = custom_xywh
        self.xywh_only = xywh_only
        self.pil = pil
        self.dst = dst
        self.save_as = save_as
        if lazy:
            self.lazy_patch()
        self.i = 0

    def __len__(self):
        return self.nb_valid_patches

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.nb_valid_patches:
            raise StopIteration
        x = self.__getitem__(self.i)
        self.i += 1
        return x

    def __getitem__(self, index):
        if 0 <= index < len(self):
            x, y, w, h = self.valid_patches[index]
            if self.xywh_only:
                return (x, y, w, h)
            tile, (x, y, w, h) = self.get_tile(x, y, w, h)
            return tile, (x, y, w, h)
        else:
            raise IndexError("Index out of range")

    def lazy_patch(self):
        if isinstance(self.custom_xywh, np.ndarray):
            if self.custom_xywh.shape[1] != 4:
                raise ValueError(
                    "custom_xywh must be a (N, 4) array of int [[x, y, w, h]]"
                )
            self.nb_valid_patches, self.valid_patches = (
                len(self.custom_xywh),
                self.custom_xywh,
            )
        else:
            self.nb_valid_patches, self.valid_patches = self.patch_sampling()
        if self.dst:
            self.patch_path = self.save_patch(self.dst, self.save_as)
        return 0

    def get_true_downsample(self, level):
        dim_0 = self.slide.dimensions
        dim_level = self.slide.level_dimensions[level]
        down_x, down_y = (dim_0[0] / dim_level[0], dim_0[1] / dim_level[1])
        return max(down_x, down_y)

    def get_seg_mask(self, margin: tuple | None = (32, 32)):
        if not self.mask_downsample:
            self.mask_downsample = 1.0
            self.level_mask = self.level_target
        else:
            self.level_mask = self.slide.get_best_level_for_downsample(
                self.mask_downsample
            )

        if not self.mask_tolerance:
            self.mask_tolerance = 0

        if not margin:
            shape_mask = (
                self.slide.level_dimensions[self.level_mask][1],
                self.slide.level_dimensions[self.level_mask][0],
            )
            margin = (shape_mask[0] // 30, shape_mask[1] // 30)
        mask = make_auto_mask(self.slide, self.level_mask, margin=margin)
        return mask

    def patch_sampling(self):
        mask = self.get_seg_mask()
        min_row, min_col, max_row, max_col = 0, 0, *mask.shape
        point_start_mask = min_row, min_col
        point_end_mask = max_row, max_col

        shape_at_level = (
            self.slide.level_dimensions[self.level_target][1],
            self.slide.level_dimensions[self.level_target][0],
        )
        point_start = get_x_y_to(point_start_mask, mask.shape, shape_at_level)
        point_end = get_x_y_to(point_end_mask, mask.shape, shape_at_level)

        # does overlapping works well
        # during init check is patch_shape_no_margin != (0, 0) -> 2 * overlap_target < patch_size_target -> in that case no marging
        patch_shape_no_margin = (
            self.patch_size_target - self.overlap_target,
            self.patch_size_target - self.overlap_target,
        )
        grid_coord = grid_blob(point_start, point_end, patch_shape_no_margin)

        return self.get_valid_patches(mask, grid_coord)

    def get_valid_patches(self, mask, grid_coord):
        shape_at_level = (
            self.slide.level_dimensions[self.level_target][1],
            self.slide.level_dimensions[self.level_target][0],
        )
        shape_mask = mask.shape
        patches_at_level = []
        patch_size_mask = get_size_to(
            (self.patch_size_target, 0), self.level_target, self.level_mask
        )[0]
        radius = np.array([max(patch_size_mask // 2, 1), max(patch_size_mask // 2, 1)])
        for coord in grid_coord:
            coord_mask = get_x_y_to(coord, shape_at_level, shape_mask)
            point_cent_mask = [
                coord_mask + radius,
                shape_mask - np.array([1, 1]) - radius,
            ]
            point_cent_mask = np.array(point_cent_mask).min(axis=0)
            if mask_percentage(mask, point_cent_mask, radius, self.mask_tolerance):
                still_add = True
                if ((coord_mask + radius) != point_cent_mask).any():
                    still_add = False
                if still_add:
                    valid_patch = [
                        coord[1],
                        coord[0],
                        self.patch_size_target,
                        self.patch_size_target,
                    ]
                    patches_at_level.append(valid_patch)  # x, y, w, h
        nb_valid_patches = len(patches_at_level)
        return nb_valid_patches, patches_at_level

    def get_tile(self, x, y, w, h):
        if self.pil:
            tile = self.slide.read_region(
                location=(x, y), level=self.level_target, size=(w, h), numpy=False
            ).convert("RGB")
        else:
            tile = self.slide.read_region(
                location=(x, y), level=self.level_target, size=(w, h)
            )
        return tile, (x, y, w, h)

    def get_thumbnail(self, size, numpy: bool = False):
        thumbnail = self.slide.get_thumbnail(size)
        if numpy:
            thumbnail = np.array(thumbnail)[:, :, :3]
        return thumbnail

    def visualize_tissue_seg(self, size: tuple, save_seg: str, show: bool = False):
        mask = self.get_seg_mask().astype(np.uint8) * 255
        mask_height, mask_width = mask.shape
        if mask_width > mask_height:
            thumbnail_width = size[0]
            thumbnail_height = int(size[1] * mask_height / mask_width)
        else:
            thumbnail_height = size[1]
            thumbnail_width = int(size[0] * mask_width / mask_height)
        thumbnail_dim = (thumbnail_width, thumbnail_height)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(mask, thumbnail_dim, interpolation=cv2.INTER_LINEAR)

        # Add annotations
        text_x_offset = int(thumbnail_width * 0.03)
        text_y_spacing = 25  # Vertical spacing between lines of text

        text_box_height = 150
        text_box_width = 300
        mask[:text_box_height, :text_box_width] = (
            mask[:text_box_height, :text_box_width] * 0.25
        ).astype(np.uint8)
        cv2.putText(
            mask,
            f"Tissue mask",
            (text_x_offset, text_y_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            mask,
            f"width={self.width}, height={self.height}",
            (text_x_offset, text_y_spacing * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            mask,
            f"mpp={self.pixel_size_0}, mag={self.mag_0}x",
            (text_x_offset, text_y_spacing * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            mask,
            f"downsample={self.mask_downsample} from {self.mag_0}x",
            (text_x_offset, text_y_spacing * 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # Save visualization
        os.makedirs(save_seg, exist_ok=True)
        cut_path = os.path.join(save_seg, f"{self.slide.name}.jpg")
        Image.fromarray(mask).save(cut_path)
        if show:
            _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
            ax.set_axis_off()
            ax.imshow(mask, aspect="equal")

        return cut_path

    def visualize_cut(self, size: tuple, save_cut: str, show: bool = False):
        thumbnail = self.get_thumbnail(size, numpy=True)
        thumbnail_height, thumbnail_width, _ = thumbnail.shape
        downsample_factor = (
            self.slide.level_dimensions[self.level_target][0] / thumbnail_width
        )
        thumbnail_patch_size = max(1, int(self.patch_size_target / downsample_factor))
        # Draw rectangles for patches
        for x, y, _, _ in self.valid_patches:
            x, y = get_x_y_to(
                (x, y),
                self.slide.level_dimensions[self.level_target],
                (thumbnail_width, thumbnail_height),
                integer=True,
            )
            thickness = max(1, thumbnail_patch_size // 16)
            thumbnail = cv2.rectangle(
                thumbnail,
                (x, y),
                (x + thumbnail_patch_size, y + thumbnail_patch_size),
                (255, 0, 0),
                thickness,
            )

        # Add annotations
        text_x_offset = int(thumbnail_width * 0.03)
        text_y_spacing = 25  # Vertical spacing between lines of text

        text_box_height = 150
        text_box_width = 300

        # Define the text box color and alpha transparency
        text_box_color = (204, 139, 189)  # Dark gray
        alpha = 0.25  # Transparency level (0 = fully transparent, 1 = fully opaque)
        # Extract the region of interest (ROI) from the original thumbnail
        roi = thumbnail[:text_box_height, :text_box_width].copy()
        # Create an overlay of the same size as the ROI, filled with the background color
        text_box = np.full_like(roi, text_box_color, dtype=np.uint8)
        overlay = cv2.addWeighted(text_box, 1 - alpha, roi, alpha, 0)
        thumbnail[:text_box_height, :text_box_width] = overlay

        cv2.putText(
            thumbnail,
            f"{self.nb_valid_patches} patches",
            (text_x_offset, text_y_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumbnail,
            f"width={self.width}, height={self.height}",
            (text_x_offset, text_y_spacing * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumbnail,
            f"mpp={self.pixel_size_0}, mag={self.mag_0}x",
            (text_x_offset, text_y_spacing * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumbnail,
            f"patch={self.patch_size_target} w. overlap={self.overlap_target} at {self.mag_target}x",
            (text_x_offset, text_y_spacing * 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumbnail,
            f"tissue tolerence={self.mask_tolerance * 100}%",
            (text_x_offset, text_y_spacing * 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Save visualization
        os.makedirs(save_cut, exist_ok=True)
        cut_path = os.path.join(save_cut, f"{self.slide.name}.jpg")
        Image.fromarray(thumbnail).save(cut_path)

        if show:
            _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
            ax.set_axis_off()
            ax.imshow(thumbnail, aspect="equal")

        return cut_path

    def save_patch(self, dst: str | None = None, save_as: str | None = "h5"):
        if save_as == "h5":
            coords = {"coords": np.array(self.valid_patches)}
            attributes = {
                "patch_size": self.patch_size_target,  # Reference frame: patch_level
                "patch_size_level0": self.patch_size_0,
                "target_magnification": self.mag_target,
                "traget_level": self.level_target,
                "target_mpp": self.pixel_size_target,
                "level0_magnification": self.mag_0,
                "level0_mpp": self.pixel_size_0,
                "overlap": self.overlap_target,
                "level0_overlap": self.overlap_0,
                "tissu_thr": self.mask_tolerance,
                "name": self.slide.name,
                "savetodir": dst,
            }
            # Save the assets and attributes to an hdf5 file
            os.makedirs(os.path.join(dst, "patches"), exist_ok=True)
            patch_file = os.path.join(
                dst, "patches", f"{self.slide.name}_patches.{save_as}"
            )
            save_h5(
                patch_file, assets=coords, attributes={"coords": attributes}, mode="w"
            )
        else:
            raise ValueError(f"Invalid save_as: {save_as}. Only h5 is supported.")
        return patch_file


# class TileSampler


# class TileEncoder
class TileEncoder:
    def __init__(
        self,
        slide,
        tile_encoder,
        coords_path: str,
        device: Optional[str] = "cuda",
        num_workers: Optional[int] = 0,
        batch_max: Optional[int] = 512,
        dst: Optional[str] = None,
        save_as: Optional[str] = "h5",
        feat_only: Optional[bool] = True,
        lazy: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ):
        self.slide = slide
        self.encoder = tile_encoder
        self.precision = tile_encoder.precision
        self.transforms = tile_encoder.eval_transforms
        self.tile_attr, self.tile_coords = read_h5_coords(coords_path)
        self.get_patch_attributes()
        self.device = device
        self.num_workers = num_workers
        self.batch_max = batch_max
        self.dst = dst
        self.save_as = save_as
        self.feat_only = feat_only
        self.verbose = verbose
        self.i = 0
        if lazy:
            self.lazy_encoder()

    def __len__(self):
        return self.nb_features

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.nb_features:
            raise StopIteration
        x = self.__getitem__(self.i)
        self.i += 1
        return x

    def __getitem__(self, index):
        if 0 <= index < len(self):
            feat = self.features[index]
            if self.feat_only:
                return feat
            else:
                x, y, w, h = self.tile_coords[index]
                tile = self.slide.get_tile(x, y, w, h)
                return tile, feat, (x, y, w, h)
        else:
            raise IndexError("Index out of range")

    def progress_bar(self, lenght, verbose):
        progress = tqdm(
            desc=f"{self.slide.name} enc with {self.encoder.enc_name}",
            total=lenght,
            unit="batch",
            initial=0,
            leave=False,
            disable=not verbose,
        )
        return progress

    def lazy_encoder(self):
        if self.name:
            assert self.name == self.slide.name, "tiles are patched in another slide"
        self.nb_features, self.features = self.extract_patch_features()
        if self.dst:
            self.feat_path = self.save_features(self.dst, self.save_as)

    def get_patch_attributes(self):
        try:
            self.patch_size = self.tile_attr.get("patch_size", None)
            self.patch_size_0 = self.tile_attr.get("patch_size_level0", None)
            self.mag_target = self.tile_attr.get("target_magnification", None)
            self.mag_0 = self.tile_attr.get("level0_magnification", None)
            self.level_target = self.tile_attr.get("traget_level", None)
            self.overlap_target = self.tile_attr.get("overlap", None)
            self.overlap_0 = self.tile_attr.get("level0_overlap", None)
            self.tissu_thr = self.tile_attr.get("tissu_thr", None)
            self.name = self.tile_attr.get("name", None)
            if None in (self.patch_size, self.mag_0, self.mag_target):
                raise KeyError("Missing attributes in patch file.")
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read patch file attributes ({str(e)}).")
            # todo work around to get patch info

    @torch.inference_mode()
    def extract_patch_features(self):
        patcher = SlidePatcher(
            slide=self.slide,
            mag_0=self.mag_0,
            mag_target=self.mag_target,
            patch_size=self.patch_size,
            overlap=self.overlap_target,
            mask_tolerance=self.tissu_thr,
            custom_xywh=self.tile_coords,
            xywh_only=False,
            pil=True,
        )

        dataset = PatchSampler(patcher=patcher, transform=self.transforms)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_max,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        progress = self.progress_bar(dataloader.__len__(), verbose=self.verbose)
        features = []
        for batch_tiles, _ in dataloader:
            batch_tiles = batch_tiles.to(self.device)
            with torch.autocast(
                device_type=self.device,
                dtype=self.precision,
                enabled=(self.precision != torch.float32),
            ):
                batch_features = self.encoder(batch_tiles)

                free_mem, _ = torch.cuda.mem_get_info()
                progress.set_postfix_str(
                    f"Free memory = {free_mem / (1024 ** 2):.2f} MB", refresh=True
                )
                progress.update()
            features.append(batch_features.cpu().numpy())
        progress.clear()

        # Concatenate features
        features = np.concatenate(features, axis=0)
        return features.shape[0], features

    def cluster_patch_features(
        self, pcs: int | None = None, neighbors: int = 20, resolution: float = 0.2
    ):
        # assert is sefl.features not nan
        features_clustered = ad.AnnData(
            X=self.features,
            obs=pd.DataFrame(self.tile_coords, columns=["x", "y", "w", "h"]),
        )
        sc.pp.neighbors(
            features_clustered, n_neighbors=neighbors, n_pcs=pcs, use_rep="X"
        )
        sc.tl.umap(features_clustered)  # Compute UMAP
        sc.tl.leiden(features_clustered, resolution=resolution)
        return features_clustered

    def visualize_feat(
        self,
        pcs: int | None = None,
        neighbors: int = 20,
        resolution: float = 0.2,
        cmap: str = "gist_rainbow",
        res_to_view: int = 4,
        save_cluster: str | None = None,
        show: bool = False,
    ):

        clusters = self.cluster_patch_features(pcs, neighbors, resolution)
        clusters.obs["leiden_int"] = clusters.obs["leiden"].astype({"leiden": "int32"})
        nc = len(clusters.obs["leiden"].unique().tolist())
        hues = np.linspace(0, 1, nc, endpoint=False)  # Evenly spaced hues
        color_map = plt.cm.get_cmap(cmap)
        colors = color_map(hues)
        np.random.shuffle(colors)
        palette = {str(i): colors[i, :] for i in range(nc)}
        vmin, vmax = clusters.obs["leiden_int"].min(), clusters.obs["leiden_int"].max()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
        ax = ax.ravel()
        visualise_tile_feat(
            self.slide,
            clusters.obs,
            name="leiden_int",
            analyse_level=self.level_target,
            res_to_view=res_to_view,
            plot_args={
                "color": None,
                "size": (6, 6),
                "title": None,
                "alpha": 0.6,
                "cmap": ListedColormap(colors),
                "vmin": vmin,
                "vmax": vmax,
            },
            loc=None,
            ax=ax[0],
            show=False,
        )

        sc.pl.umap(
            clusters,
            color="leiden",
            title=f"UMAP of {self.encoder.enc_name} feat colored by Leiden Clusters",
            palette=palette,
            ax=ax[1],
            show=False,
            legend_loc="right margin",
        )
        if show:
            plt.show
        # Save visualization
        if save_cluster:
            os.makedirs(save_cluster, exist_ok=True)
            cluster_path = os.path.join(save_cluster, f"{self.slide.name}.jpg")
            fig.savefig(cluster_path)

        return cluster_path

    def save_features(self, dst, save_as):
        # Save the features to disk
        features_dir = os.path.join(dst, f"features_{self.encoder.enc_name}")
        os.makedirs(features_dir, exist_ok=True)
        if save_as == "h5":
            features_path = os.path.join(features_dir, f"{self.name}.{save_as}")
            assets = {"features": self.features, "coords": self.tile_coords}
            attributes = {
                "features": {
                    "encoder": self.encoder.enc_name,
                    "name": self.name,
                    "dst": features_dir,
                },
                "coords": self.tile_attr,
            }
            save_h5(
                features_path,
                assets=assets,
                attributes=attributes,
                mode="w",
            )
        else:
            raise ValueError(f"Invalid save_as: {save_as}. Only h5 is supported.")
        return features_path


# add possibility to generate random biopsies
class PatchSampler(Dataset):
    """Dataset from a WSI patcher to read tiles"""

    def __init__(self, patcher, transform):
        self.patcher = patcher
        self.transform = transform

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, index):
        tile, (x, y, w, h) = self.patcher[index]
        if self.transform:
            tile = self.transform(tile)
        return tile, (x, y, w, h)


class EncodingSampler:
    def __init__(self, feat_path, n_samples=0):
        """
        Sampler to get feat from h5 file
        """
        self.feat_path = feat_path
        self.name, _ = os.path.splitext(os.path.basename(feat_path))
        self.n_samples = n_samples
        self.attributes, self.features = self.read_h5(feat_path)

    def __len__(self):
        return self.features.shape[0]

    def get_feat(self):
        sample = getattr(self.sampler + "_sampler")(self.n_samples)
        print(sample)
        feat = self.features[sample]
        return sample, feat

    def random_sampler(self, n_samples):
        indices = torch.randint(0, self.__len__(), (n_samples,))
        return indices

    def all_sampler(self, n_samples):
        indices = list(range(self.__len__()))
        return indices

    def random_strict_sampler(self, n_samples):
        if self.__len__() >= n_samples:
            indices = self.random_sampler(n_samples)
        else:
            indices = self.all_sampler(n_samples)
        return indices

    def read_h5(self, path):
        with h5py.File(path, "r") as f:
            attrs = dict(f["features"].attrs)
            feats = f["features"][:]
        return attrs, feats
