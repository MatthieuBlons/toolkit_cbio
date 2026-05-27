# General libraries
import os
from tqdm import tqdm
import warnings
from typing import Optional
from functools import partial
import h5py

# Data libraries
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import PIL
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

# ML libraries
from sklearn.preprocessing import MinMaxScaler

# Plotting libraries
import matplotlib.pyplot as plt

# Project modules
from slide.utils import (
    grid_blob,
    save_h5,
    read_h5_coords,
    read_h5_features,
    get_size_to,
    get_x_y_to,
)
from slide.image_processing import (
    mask_percentage,
    compute_otsu_mask,
    compute_luminosity_mask,
    hog_selection_torch,
    color_filter_selection_torch,
)
from draw.palette import mosaic
from slide.draw import (
    visualise_tile_feat,
    visualize_cut,
    visualize_tissue_seg,
    visualise_tile_rgb,
)
from slide.reader import OpenWSI
from bioformat.reader import OpenOME


class SlidePatcher:
    """SlidePatcher is use to extract tissue patches from a WSI."""

    # Creating a dictionary to store the tissue mask selection strategies mapping with the corresponding function
    mask_strategy_mapping = {
        "luminosity": compute_luminosity_mask,
        "otsu": compute_otsu_mask,
    }
    # Creating a dictionary to store the tile selection strategies mapping with the corresponding function
    tile_selection_mapping = {"hog": hog_selection_torch}

    def __init__(
        self,
        slide: OpenWSI | OpenOME,
        pixel_size_0: int = None,
        pixel_size_target: int = None,
        mag_0: int = None,
        mag_target: int = 20,
        patch_size: int = 256,
        overlap: int = 0,
        mask_downsample: int = 16,
        margin: int | None = None,
        mask_tolerance: int = None,
        mask_strategy: str = "otsu",
        custom_mask: np.ndarray = None,
        custom_xywh: np.ndarray = None,
        overwrite: bool = True,
        selection_strategy: str = None,
        color_thresh: tuple[int, int, int] = None,
        stat_fct: str = "channel_std",
        q: float = 0.2,
        remove_background: bool = True,
        background_thresh: int | tuple[int, int, int] = 245,
        background_average: bool = True,
        local_average: bool = True,
        filter_size: tuple[int, int] | int = 16,
        filter_all: bool = False,
        aggregate: str = "mean",
        xywh_only: bool = True,
        pil: bool = False,
        dst: str = None,
        save_as: str = "h5",
        lazy: bool = True,
        *args,
        **kwargs,
    ):
        """
        Build an instance of SlidePatcher.

        Args:
            slide:                  openslide.OpenSlide, slide to be analyzed.
            pixel_size_0:           float, size of the pixel at level 0, default=None.
            pixel_size_target:      float, size of the pixel at the target level, default=None.
            mag_0:                  int, magnification at level 0, default=None.
            mag_target:             int, magnification at the target level, default=20.
            patch_size:             int, size of the patch, default=256.
            overlap:                int, number of overlapping pixels between patches, default=0.
            mask_downsample:        int, downsampling factor for the patch, default=32.
            mask_tolerance:         int, tolerance for the tissue segmentation mask, default=0.5.
            mask_strategy:          str, strategy to use to generate tissue mask, default=otsu.
            custom_xywh:            np.ndarray, array containing the custom coordinates, default=None.
            xywh_only:              bool, whether to only compute the coordinates, default=True.
            pil:                    bool, whether to use a pil image, default=False.
            dst:                    str, path to save the directory where to save the outputs, default=None, meaning the
                                    outputs will not be saved.
            save_as:                str, name of the extension for the saving the patches' file, default='h5'.
            lazy:                   bool, whether to use lazy patching, default=True.
            selection_strategy:     str, strategy for the selecting the tiles, default=None, meaning they will not be
                                    filtered.
            color_thresh:           tuple of int, color threshold for each channel (R,G,B), default=None, meaning no
                                    color filtering is applied to filter tiles.
            stat_fct:               str, statistic to compute and compare to the color threshold, default='channel_std'.
            q:                      float, quantile to compute for tile filtering based on color, default=0.2.
            remove_background:      bool, whether to remove the background before computing the color statistic threshold,
                                    default=True.
            background_thresh:      tuple of int or int, color threshold for detecting background for each channel (R,G,B),
                                    default=245.
            background_average:     bool, whether to detect the background on the average pixels or the pixels, default=True.
            local_average:          bool, whether to compute the local average before computing the statistic value,
                                    default=True.
            filter_size:            tuple of int or int, size of the patches to perform local averaging before computing the
                                    statistic, default=16.
            filter_all:             bool, whether the criterion on the color threshold has to be met for all channel or
                                    only one, default=False.
            aggregate:              str, aggregate function to aggregate the channel std over the pixels, default='mean'.
            args:                   list or tuple, additional positional arguments to pass to the tile selection function.
            kwargs:                 dict, additional keyword arguments to pass to the tile selection function.

        """
        # Storing useful attributes
        self.slide = slide
        self.width, self.height = slide.dimensions
        self.patch_size_target = patch_size
        self.mag_target = mag_target
        self.overlap_target = overlap
        self.custom_xywh = custom_xywh
        self.overwrite = overwrite
        self.xywh_only = xywh_only
        self.pil = pil
        self.dst = dst
        self.save_as = save_as
        self.selection_strategy = selection_strategy
        self.patch_path = None
        self.valid_patches = None
        self.nb_valid_patches = None
        self.idx_selected = None

        # Storing color filtering args
        self.color_thresh = color_thresh
        self.stat_fct = stat_fct
        self.q = q
        self.remove_background = remove_background
        self.background_thresh = background_thresh
        self.background_average = background_average
        self.local_average = local_average
        self.filter_size = filter_size
        self.filter_all = filter_all
        self.aggregate = aggregate

        # Parsing source image default args using the slide attributes
        self.mag_0 = mag_0 if mag_0 is not None else slide.magnification
        self.pixel_size_0 = pixel_size_0 if pixel_size_0 is not None else slide.mpp

        # Parsing tolerance for tissue mask default args
        self.mask_tolerance = mask_tolerance if mask_tolerance is not None else 0

        # Computing target level's associated values
        downsample = self.mag_0 / self.mag_target
        self.pixel_size_target = pixel_size_target
        if pixel_size_target is None:
            self.pixel_size_target = self.pixel_size_0 * downsample
        self.level, self.downsample_level, self.resize_factor = (
            self.slide.get_best_level_for_downsample(downsample)
        )

        # Parsing mask downsample and get associated level
        if mask_downsample is None:
            self.mask_downsample = 1.0
            self.level_mask = self.level
        else:
            self.mask_downsample = mask_downsample
            self.level_mask, _, _ = self.slide.get_best_level_for_downsample(
                self.mask_downsample
            )
        self.margin = (margin, margin) if margin is not None else None
        self.mask_strategy = mask_strategy
        self.mask_selection = self.get_mask_strategy(
            mask_strategy=self.mask_strategy, *args, **kwargs
        )
        self.custom_mask = custom_mask

        # Computing patch size and overlap in original dimensions
        self.patch_size_level = round(self.patch_size_target / self.resize_factor)
        self.overlap_level = round(self.overlap_target / self.resize_factor)

        # Getting the tile selection mapping function
        if self.selection_strategy is not None:
            self.tile_selection = self.get_selection_strategy(
                selection_strategy=self.selection_strategy, *args, **kwargs
            )
        else:
            self.tile_selection = None

        # Performing lazy patching
        if lazy:
            self.lazy_patch()

        # Initialization of a counter for filtering valid patches
        self.i = 0

    def __len__(self) -> int:
        """
        Return the number of valid patches in the slide.

        Returns:
            n_patches: int, number of valid patches in the slide.

        """
        return self.nb_valid_patches

    def __iter__(self):
        """
        Reinitialize the counter for the valid tiles index.
        """
        self.i = 0
        return self

    def __next__(
        self,
    ) -> tuple[PIL.Image.Image, tuple[int, int, int, int]] | tuple[int, int, int, int]:
        """
        Return the next valid patch in the slide.

        Returns:
            patch:      PIL.Image.Image, the next valid patch in the slide if not `xywh_only`.
            coords:     tuple of int, the coordinates (x,y,w,h) of the next valid patch in the slide.

        """

        # Check if it is a valid patch
        if self.i >= self.nb_valid_patches:
            raise StopIteration

        # Get the patch and update counter
        x = self.__getitem__(self.i)
        self.i += 1

        # Return the patch
        return x

    def __getitem__(
        self, index: int
    ) -> tuple[PIL.Image.Image, tuple[int, int, int, int]] | tuple[int, int, int, int]:
        """
        Get the patch at the given index and return its image and coordinates.

        Args:
            index:      int, the index of the patch to be retrieved.

         Returns:
            patch:      PIL.Image.Image, the patch at the given index in the slide if not `xywh_only`.
            coords:     tuple of int, the coordinates (x,y,w,h) of the patch at the given index in the slide.

        """
        # Check if it is a valid patch
        if 0 <= index < len(self):

            # Retrieve coordinates
            coords = self.valid_patches[index]

            # Retrieve the tile
            if not self.xywh_only:
                res = (self.get_tile(*coords), coords)
            else:
                res = coords

            # Returning the results
            return res

        # Otherwise raise an error
        else:
            raise IndexError("Index out of range")

    def get_mask_strategy(self, mask_strategy: str, *args, **kwargs) -> callable:
        """
        Get the corresponding tissue masking function based on the given mask strategy and given the additional
        arguments provided in args and kwargs.
        Args:
            mask_strategy:     str, the selection strategy to use.

        Returns:
            mask_func:         callable, the function which will be applied to the selected tiles.

        """
        # Fetch the corresponding selection strategy
        if mask_strategy.lower() in SlidePatcher.mask_strategy_mapping.keys():
            return partial(
                SlidePatcher.mask_strategy_mapping[mask_strategy.lower()],
                *args,
                **kwargs,
            )

        # Otherwise raise an error
        else:
            raise NotImplementedError(
                "mask strategy {} is not implemented.".format(mask_strategy)
                + f"\nPlease choose a valid strategy from: {' ,'.join(SlidePatcher.mask_strategy_mapping.keys())}."
            )

    def get_selection_strategy(
        self, selection_strategy: str, *args, **kwargs
    ) -> callable:
        """
        Get the corresponding tile selection function based on the given selection strategy and give the additional
        arguments provided in args and kwargs.
        Args:
            selection_strategy:     str, the selection strategy to use.

        Returns:
            selection_func:         callable, the function which will be applied to the selected tiles.

        """
        # Fetch the corresponding selection strategy
        if selection_strategy.lower() in SlidePatcher.tile_selection_mapping.keys():
            return partial(
                SlidePatcher.tile_selection_mapping[selection_strategy.lower()],
                *args,
                **kwargs,
            )

        # Otherwise raise an error
        else:
            raise NotImplementedError(
                "Tile selection strategy {} is not implemented.".format(
                    selection_strategy
                )
                + f"\nPlease choose a valid strategy from: {' ,'.join(SlidePatcher.tile_selection_mapping.keys())}."
            )

    def lazy_patch(self):
        """
        Perform lazy patching using the attributes stored in the SlidePatcher for the tissue segmentation parameters and
        the patching of the slide.
        """

        # Check and load the custom coordinates
        if isinstance(self.custom_xywh, np.ndarray):
            if self.custom_xywh.shape[1] != 4:
                raise ValueError(
                    "custom_xywh must be a (n, 4) array of int [[x, y, w, h]]"
                )
            self.nb_valid_patches, self.valid_patches = (
                len(self.custom_xywh),
                self.custom_xywh,
            )

        # Otherwise, segment and patch the slide
        else:
            self.nb_valid_patches, self.valid_patches = self.patch_sampling()
            # Saving the outputs
        if (self.dst is not None) & self.overwrite:
            self.patch_path = self.save_patch(self.dst, self.save_as)

    def get_seg_mask(self, margin: tuple[int, int] | None = None) -> np.array:
        """
        Perform the tissue segmentation task on the slide and obtain the corresponding segmentation mask.

        Args:
            margin:     tuple of int, the margin size of the segmentation mask on each dimension in number of pixels
                        at level_mask.

        Returns:
            seg_mask:    np.array, array containing the segmentation mask on the slide.

        """
        if self.custom_mask is not None:
            return self.custom_mask

        # Parsing margin default args
        if margin is None:
            shape_mask = (
                self.slide.level_dimensions[self.level_mask][1],
                self.slide.level_dimensions[self.level_mask][0],
            )
            margin = (shape_mask[0] // 100, shape_mask[1] // 100)

        # Computing segmentation mask
        mask = self.mask_selection(self.slide, self.level_mask, margin=margin)

        # Returning segmentation mask
        return mask, margin

    def patch_sampling(
        self,
        select_tiles: bool = None,
        color_filter: bool = None,
        store_idx: bool = True,
    ) -> tuple[int, list[list[int]]]:
        """
        Patchify the slide by first, performing the tissue segmentation task and then extracting valid patches on the
        tissue areas of the slide. Additional patch filtering using the tile selection method specified in the instance
        attributes.

        Args:
            select_tiles:       bool, whether to perform tile selection, default=None, meaning it will infer the
                                behaviour from instance's attributes.
            color_filter:       bool, whether to filter the patches based on the colors, default=None, meaning it
                                will infer the behaviour from instance's attributes.
            store_idx:          bool, whether to store the selected patches indices in the instance, default=True.

        Returns:
            nb_valid_patches:   int, the number of valid patches in the slide.
            valid_patches:      list of lists of int, list containing the valid patches' coordinates (x,y,w,h) in
                                the slide.

        """

        # Parsing default args
        if select_tiles is None:
            select_tiles = True if self.tile_selection is not None else False
        if color_filter is None:
            color_filter = True if self.color_thresh is not None else False

        # Tissue segmentation task and retrieving useful coordinates and sizes at the
        # original resolution for patching

        mask, margin = self.get_seg_mask(self.margin)

        min_row, min_col, max_row, max_col = (
            margin[0],
            margin[1],
            mask.shape[0] - margin[0],
            mask.shape[1] - margin[1],
        )

        point_start_mask = min_row, min_col
        point_end_mask = max_row, max_col

        # Converting the coordinates and size to the target level resolution
        shape_at_level = (
            self.slide.level_dimensions[self.level][1],
            self.slide.level_dimensions[self.level][0],
        )
        point_start = get_x_y_to(point_start_mask, mask.shape, shape_at_level)
        point_end = get_x_y_to(point_end_mask, mask.shape, shape_at_level)

        # Creating the grid of the patches, removing the overlapping margin
        patch_shape_no_margin = (
            self.patch_size_level - self.overlap_level,
            self.patch_size_level - self.overlap_level,
        )
        grid_coord = grid_blob(point_start, point_end, patch_shape_no_margin)

        # Obtain the valid patches from tissue segmentation and grid coordinates
        nb_valid_patches, valid_patches = self.get_valid_patches(mask, grid_coord)

        # Perform tile selection and color filtering of the tiles
        if select_tiles or color_filter:
            nb_valid_patches, valid_patches, idx_selected = self.select_patches(
                patch_list=valid_patches,
                return_idx=True,
                color_filter=color_filter,
                select_tiles=select_tiles,
            )
            if store_idx:
                self.idx_selected = idx_selected

        # Returning the valid patches (intersection of the mask on the grid and additional filtering)
        return nb_valid_patches, valid_patches

    def get_invalid_patches(
        self,
        from_tile_method: bool = False,
        from_color_filter: bool = True,
        verbose: bool = True,
    ) -> tuple[int, list[list[int]]]:
        """
        Get invalid patches from the tile selection and color filtering process.

        Args:
            from_tile_method:       bool, whether to get the unselected patches from tile selection, default=False.
            from_color_filter:      bool, whether to get the unselected patches from color filtering, default=True.
            verbose:                bool, whether to print information about the selection process, default=True.

        Returns:
            nb_patches:             int, number of unselected patches in the slide.
            invalid_patches:        list of lists of int, list containing the invalid patches' coordinates in the slide.

        """

        # Parsing select tiles and color filter
        color_filter, select_tiles = None, None
        if from_color_filter:
            color_filter = False
        if from_tile_method:
            select_tiles = False

        # Retrieving all patches from tissue segmentation
        _, all_patches = self.patch_sampling(
            select_tiles=select_tiles, color_filter=color_filter
        )

        # Retrieving invalid patches coordinates
        invalid_patches = np.array(all_patches)[
            ~np.isin(
                np.array(list(map(lambda x: str(x), np.array(all_patches)))),
                np.array(list(map(lambda x: str(x), np.array(self.valid_patches)))),
            )
        ].tolist()

        # Printing information
        if verbose:
            print(
                "Number of invalid patches: {}/{}".format(
                    len(invalid_patches), len(all_patches)
                )
            )

        # Returning number of invalid patches and corresponding coordinates
        return len(invalid_patches), invalid_patches

    def get_valid_patches(
        self, mask: np.ndarray, grid_coord: np.ndarray
    ) -> tuple[int, list[list[int]]]:
        """
        Get valid patches from the tissue segmentation mask and the grid coordinate computed at the target resolution.

        Args:
            mask:           np.ndarray, the tissue segmentation mask on the slide.
            grid_coord:     np.ndarray, the grid coordinates computed at the target resolution.

        Returns:
            n_patches:      int, the number of valid patches in the slide.
            valid_patches:  list of lists of int, the valid patches' coordinates (x,y,w,h) in the slide.

        """

        # Retrieving the source and target shapes
        shape_at_level = (
            self.slide.level_dimensions[self.level][1],
            self.slide.level_dimensions[self.level][0],
        )
        shape_mask = mask.shape

        # Initialization of a list to store the valid patches at the desired resolution
        patches_at_level = []

        # Computing the size of the patch in the original resolution
        patch_size_mask = get_size_to(
            size=(self.patch_size_level, 0),
            downsample_from=self.downsample_level,
            downsample_to=self.mask_downsample,
        )[0]

        # Computing radius of patch in original resolution
        radius = np.array([max(patch_size_mask // 2, 1), max(patch_size_mask // 2, 1)])

        # For each patch in the grid, check if valid patch and obtain coordinates and image at target resolution
        for coord in grid_coord:

            # Get coordinates in original resolution
            coord_mask = get_x_y_to(coord, shape_at_level, shape_mask)

            # Compute patch's centroids at original resolution
            point_cent_mask = np.array(
                [coord_mask + radius, shape_mask - np.array([1, 1]) - radius]
            ).min(axis=0)

            # Check if valid patch by computing tissue percentage
            if mask_percentage(
                mask=mask,
                point=point_cent_mask,
                radius=radius,
                mask_tolerance=self.mask_tolerance,
            ):

                # Check if entire patch is in image
                still_add = True
                if np.array((coord_mask + radius) != point_cent_mask).any():
                    still_add = False

                # If valid patch, add its coordinates to the list
                if still_add:
                    valid_patch = [
                        coord[1],
                        coord[0],
                        self.patch_size_level,
                        self.patch_size_level,
                    ]
                    patches_at_level.append(valid_patch)  # x, y, w, h

        # Return number of valid patches and the list of valid patches
        return len(patches_at_level), patches_at_level

    def filter_patch_on_color(
        self,
        img_list: list[np.ndarray] | np.ndarray,
        color_thresh: tuple[int, int, int] = None,
        stat_fct: str = None,
        q: float = None,
        remove_background: bool = None,
        background_thresh: int | tuple[int, int, int] = None,
        background_average: bool = None,
        local_average: bool = None,
        filter_size: int | tuple[int, int] = None,
        filter_all: bool = None,
        aggregate: str = None,
    ) -> list[int]:
        """
        Filter the patches stored in img list based on the color threshold with one value per channel (R,G,B).
        Args:
            img_list:               list of np.ndarray or np.ndarray, list containing all the images to be filtered.
            color_thresh:           tuple of int, the color threshold value per channel.
            stat_fct:               str, statistic to compute and compare to the color threshold, default=None.
            q:                      float, quantile to compute for tile filtering based on color, default=None.
            remove_background:      bool, whether to remove the background before computing the color statistic threshold,
                                    default=None.
            background_thresh:      tuple of int or int, color threshold for detecting background for each channel (R,G,B),
                                    default=None.
            background_average:     bool, whether to detect the background on average pixels or pixels, default=None.
            local_average:          bool, whether to compute the local average before computing the statistic value,
                                    default=None.
            filter_size:            tuple of int or int, size of the patches to perform local averaging before computing the
                                    statistic, default=None.
            filter_all:             bool, whether the criterion on the color threshold has to be met for all channel or
                                    only one, default=None.
            aggregate:              str, aggregate method to aggregate the channel std over the pixels, default='mean'.

        Returns:
            idx_selected:       list of int, the indexes of the selected patches.

        """

        # Parsing default args
        if color_thresh is None:
            color_thresh = self.color_thresh
        if stat_fct is None:
            stat_fct = self.stat_fct
        if q is None:
            q = self.q
        if remove_background is None:
            remove_background = self.remove_background
        if background_thresh is None:
            background_thresh = self.background_thresh
        if background_average is None:
            background_average = self.background_average
        if local_average is None:
            local_average = self.local_average
        if filter_size is None:
            filter_size = self.filter_size
        if filter_all is None:
            filter_all = self.filter_all
        if aggregate is None:
            aggregate = self.aggregate

        # Filtering patches based on mean color value for each channel
        return color_filter_selection_torch(
            img_list,
            color_thresh=color_thresh,
            stat_fct=stat_fct,
            q=q,
            remove_background=remove_background,
            background_thresh=background_thresh,
            background_average=background_average,
            local_average=local_average,
            filter_size=filter_size,
            filter_all=filter_all,
            aggregate=aggregate,
        )

    def select_patches(
        self,
        patch_list: list[list] | np.ndarray,
        return_idx: bool = False,
        color_filter: bool = None,
        select_tiles: bool = None,
        *args,
        **kwargs,
    ) -> tuple[int, list[list]] | tuple[int, list[list], list]:
        """
        Function to further filter the patches from the tissue segmentation mask.

        Args:
            patch_list:     list of lists or np.ndarray, list containing the patches coordinates (x,y,w,h) to be filtered.
            return_idx:     bool, whether to return the index of the patches selected, default=False.
            color_filter:   bool, whether to filter the patches on the color threshold, default=None.
            select_tiles:   bool, whether to select the tiles, default=None.

        Returns:
            nb_patches:     int, the new number of valid patches in the slide.
            valid_patches:  list of lists, the valid patches coordinates in the slide.
            idx_selected:   list of int, the index of the patches selected in the slide if return_idx=True.

        """
        # Parsing default args
        if color_filter is None:
            color_filter = True if self.color_thresh is not None else False
        if select_tiles is None:
            select_tiles = True if self.tile_selection is not None else False

        # Retrieving all valid tiles
        tiles = self.get_all_tiles(patch_list)

        # Performing tile selection using the selection strategy
        if select_tiles:
            idx_selected = self.tile_selection(img_list=tiles, *args, **kwargs)
        else:
            idx_selected = np.array(list(range(len(tiles))))

        # Performing tile selection based on color filtering
        if color_filter:
            idx_kept = self.filter_patch_on_color(
                img_list=np.array(tiles)[idx_selected]
            )
            idx_selected = idx_selected[idx_kept]

        # Updating valid patches
        nb_valid_patches = len(idx_selected)
        valid_patches = (np.array(patch_list)[idx_selected]).tolist()

        # Check if any patch selected
        if nb_valid_patches == 0:
            print(
                f"Tile selection failed, no valid patches found. Check the selection strategy parameters."
            )

        # Returning valid patches
        res = nb_valid_patches, valid_patches
        if return_idx:
            res = nb_valid_patches, valid_patches, idx_selected
        return res

    def get_tile(self, x: int, y: int, w: int, h: int) -> PIL.Image.Image | np.ndarray:
        """
        Get the tile at the given coordinates in the slide at the original resolution.

        Args:
            x:      int, the x coordinate of the tile.
            y:      int, the y coordinate of the tile.
            w:      int, the width of the tile.
            h:      int, the height of the tile.

        Returns:
            tile:   PIL.Image.Image or np.ndarray, the tile retrieved at the given coordinates in the slide.
                    Its format will match the format of the slide.

        """

        # Get the tile from the slide's PIL image
        if self.pil:
            tile = self.slide.read_region(
                location=(x, y), level=self.level, size=(w, h), numpy=False
            ).convert("RGB")
            tile = tile.resize((self.patch_size_target, self.patch_size_target))

        # Get the tile from the slide's array
        else:
            tile = self.slide.read_region(
                location=(x, y), level=self.level, size=(w, h), numpy=True
            )
            tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))[
                :, :, :3
            ]

        # Returning the tile
        return tile

    def get_all_tiles(
        self, patch_list: list[list[int]] | None = None
    ) -> list[PIL.Image.Image] | list[np.ndarray]:
        """
        Get all the tile in patch_list at once at the original resolution.

        Args:

        Returns:
            tiles:   list of PIL.Image.Image or np.ndarray, all tiles retrieved in the slide.
            Its format will match the format of the slide.

        """
        if patch_list is None:
            patch_list = self.valid_patches

        # Get the tile from the slide's PIL image
        if self.pil:
            whole = self.slide.read_whole(self.level, numpy=False).convert("RGB")
            tiles = [
                whole.crop((x, y, x + w, y + h)).resize(
                    (self.patch_size_target, self.patch_size_target)
                )
                for (x, y, w, h) in self.valid_patches
            ]

        # Get the tile from the slide's array
        else:
            whole = self.slide.read_whole(self.level, numpy=True)
            tiles = [
                cv2.resize(
                    whole[y : y + h, x : x + w],
                    (self.patch_size_target, self.patch_size_target),
                )[:, :, :3]
                for (x, y, w, h) in self.valid_patches
            ]

        # Returning the tile
        return tiles

    def get_thumbnail(
        self, size: tuple[int, int], numpy: bool = False
    ) -> np.ndarray | PIL.Image.Image:
        """
        Get the slide's thumbnail at the given size, with the corresponding resolution.

        Args:
            size:       tuple of int, the desired size of slide's thumbnail.
            numpy:      bool, whether to return numpy or PIL.Image.

        Returns:
            thumbnail:  np.ndarray or PIL.Image, the thumbnail of the slide at the specified size.

        """
        thumbnail = self.slide.get_thumbnail(size)
        if numpy:
            thumbnail = np.array(thumbnail)[:, :, :3]
        return thumbnail

    def visualize_tissue_seg(
        self,
        size: tuple[int, int],
        save_seg: str = None,
        show: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Visualize the tissue segmentation mask.

        Args:
            size:           tuple of int, the size of the image of the slide's tissue segmentation mask.
            save_seg:       str, the path to the directory where to save the segmentation mask, default=None.
            show:           bool, whether to show the visualization of the tissue segmentation mask, default=False.

        Returns:
            seg_path:       str, the path to the segmentation mask including the file's name.
                            Returns None if the segmentation mask was not saved.

        """

        # Computing segmentation mask
        mask, _ = self.get_seg_mask(self.margin)
        mask = mask.astype(np.uint8) * 255
        # Creating annotations
        slide_attrs = {
            "size": (self.width, self.height),
            "mpp": self.pixel_size_0,
            "mag_0": self.mag_0,
            "downsample": self.mask_downsample,
            "strategy": self.mask_strategy,
        }

        # Plotting the tissue segmentation mask
        seg_path = visualize_tissue_seg(
            mask=mask,
            size=size,
            slide_name=self.slide.name,
            slide_attrs=slide_attrs,
            save_seg=save_seg,
            show=show,
            *args,
            **kwargs,
        )

        # Returning path to the saved image
        return seg_path

    def visualize_cut(
        self,
        size: tuple[int, int],
        save_cut: str = None,
        show: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Visualize the patches generated from the tissue segmentation mask and the grid coordinate computed at the target
        resolution.

        Args:
            size:           tuple of int, the desired size of the image on which to plot the patches.
            save_cut:       str, the path to the directory where to save the visualization of the patches, default=None.
            show:           bool, whether to show the visualization of the patches, default=False.

        Returns:
            cut_path:       str, the path to the visualization of the patches including the file name.

        """

        # Creating slide additional annotations
        slide_attrs = {
            "overlap_target": self.overlap_target,
            "overlap_level": self.overlap_level,
            "level": self.level,
            "tissue_tolerance": self.mask_tolerance,
        }

        # Plotting the patches
        cut_path = visualize_cut(
            wsi_slide=self.slide,
            coords=self.valid_patches,
            size=size,
            patch_size_target=self.patch_size_target,
            mag_target=self.mag_target,
            slide_attrs=slide_attrs,
            save_cut=save_cut,
            show_plot=show,
            *args,
            **kwargs,
        )

        # Returning path to saved image
        return cut_path

    def sample_plot_patches(
        self,
        patches: list[list[int]] | np.ndarray = None,
        n_patches: int = None,
        save_path: str = None,
        figsize=None,
        show=True,
    ):
        """

        Args:
            patches:            list of lists of int or np.ndarray, the list of the patches to be sampled, default=None.
            n_patches:          int, number of patches to be sampled, default=None. if not provided take all invalid patches
            save_path:          str, the path to the directory where to create the `invalid_patches` directory in which
                                to save the patches, default=None, meaning the figure will not be saved.
            figsize:            tuple of int, the size of the figure, default=None.
            show:               bool, whether to show the visualization of the patches, default=True.

        """

        # Getting unselected patches
        total_patches = len(patches)
        if not n_patches:
            n_patches = total_patches
        # Sampling in the invalid patches
        if n_patches < total_patches:
            i_patches = np.random.choice(
                np.arange(total_patches), n_patches, replace=False
            ).tolist()
            patches = np.array(patches)[i_patches, :].tolist()
        else:
            patches = np.array(patches).tolist()

        # Retrieving corresponding tiles image
        invalid_img = []
        for patch in patches:
            invalid_img.append(self.get_tile(*patch))

        # Plotting the images
        if figsize is None:
            if len(invalid_img) <= 16:
                figsize = (5, 5)
            elif len(invalid_img) <= 100:
                figsize = (10, 10)
            else:
                figsize = (15, 15)
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
        ax = mosaic(
            invalid_img,
            title=f"Tiles not selected in: {self.slide.name}",
            ax=ax,
        )
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, "{}.jpeg".format(self.slide.name)))
        if show:
            plt.show()
        else:
            plt.close(fig)

    def sample_plot_invalid_patches(
        self,
        n_patches: int = 36,
        save_path: str = None,
        invalid_on: str = "all",
        figsize=None,
        verbose: bool = True,
        show: bool = True,
    ):
        """
        Compute invalid patches, sample and plot some of them and save them in `invalid_patches` directory.

        Args:
            n_patches:          int, number of patches to be sampled, default=36.
            save_path:          str, the path to the directory where to create the `invalid_patches` directory in which
                                to save the patches, default=None, meaning the figure will not be saved.
            invalid_on:         str, whether plot invalid patches from the whole selection process or the tile selection
                                or the color filtering, default='all'.
            figsize:            tuple of int, the size of the figure, default=None.
            verbose:            bool, whether to print useful information about the selection process, default=True
            show:               bool, whether to show the visualization of the patches, default=True.


        """

        # Parsing which selection process we want to plot
        from_tile_method = True
        from_color_filter = True
        if invalid_on == "color":
            from_tile_method = False
        elif invalid_on == "quality":
            from_color_filter = False

        # Getting unselected patches
        nb_invalid_patches, invalid_patches = self.get_invalid_patches(
            from_tile_method=from_tile_method,
            from_color_filter=from_color_filter,
            verbose=verbose,
        )

        # Sampling and plotting the invalid patches
        if nb_invalid_patches > 0:
            self.sample_plot_patches(
                patches=invalid_patches,
                n_patches=n_patches,
                save_path=save_path,
                figsize=figsize,
                show=show,
            )

    def save_patch(self, dst: str = None, save_as: str = "h5") -> str:
        """
        Save the patches generated from the tissue segmentation mask in the given directory, specified by `dst`, in the
        format given by `save_as`.

        Args:
            dst:            str, path to the directory where to save the patches, default=None.
            save_as:        str, format of the file to save the patches, default='h5'.

        Returns:
            patch_file:    str, path to the saved patch files.

        """

        # Saving the patches as h5 file
        if save_as == "h5":

            # Retrieving the coordinates and storing useful patches attributes
            coords = {"coords": np.array(self.valid_patches)}
            attributes = {
                "magnification": self.mag_0,
                "mpp": self.pixel_size_0,
                "target_magnification": self.mag_target,
                "target_patch_size": self.patch_size_target,
                "target_mpp": self.pixel_size_target,
                "target_overlap": self.overlap_target,
                "level": self.level,
                "level_size": self.slide.level_dimensions[self.level],
                "level_patch_size": self.patch_size_level,
                "level_overlap": self.overlap_level,
                "tissue_thr": self.mask_tolerance,
                "mask": self.mask_strategy,
                "name": self.slide.name,
                "savetodir": dst,
            }

            # Creating path to the patch file
            os.makedirs(os.path.join(dst, "patches"), exist_ok=True)
            patch_file = os.path.join(
                dst, "patches", f"{self.slide.name}_patches.{save_as}"
            )

            # Save the assets and attributes to a h5 file
            save_h5(
                patch_file, assets=coords, attributes={"coords": attributes}, mode="w"
            )

        # Otherwise, raise an error
        else:
            raise ValueError(
                f"Invalid save_as argument: {save_as}. Only h5 files are supported."
            )

        # Returning the path to the patch file
        return patch_file


# could add image meta (size, mag, mpp if known)
class ImageEncoder:
    def __init__(
        self,
        imgs: list,
        tile_encoder: torch.nn.Module,
        ext: Optional[str] = None,
        device: Optional[str] = "cuda",
        num_workers: Optional[int] = 0,
        batch_max: Optional[int] = 512,
        dst: Optional[str] = None,
        save_as: Optional[str] = "h5",
        lazy: Optional[bool] = True,
        feat_only: Optional[bool] = True,
        pil: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ):
        self.imgs = imgs
        self.ext = ext
        self.encoder = tile_encoder
        self.precision = tile_encoder.precision
        self.transforms = tile_encoder.eval_transforms
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
                img = self.get_image(index, numpy=not self.pil)
                return img, feat
        else:
            raise IndexError("Index out of range")

    def get_image(self, index, numpy=True):
        img = self.imgs[index]
        if isinstance(img, np.ndarray):
            return img, None
        else:
            path = img
            img = PIL.Image.open(path)
            if not numpy:
                return img, path
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.dtype not in [np.uint8, np.float32]:
                img = np.float32(img)
            return img, path

    def progress_bar(self, lenght, verbose):
        progress = tqdm(
            desc=f"Images enc with {self.encoder.enc_name}",
            total=lenght,
            unit="batch",
            initial=0,
            leave=False,
            disable=not verbose,
        )
        return progress

    def lazy_encoder(self):
        self.nb_features, self.features = self.extract_patch_features()
        if self.dst:
            self.feat_path = self.save_features(self.dst, self.save_as)

    @torch.inference_mode()
    def extract_patch_features(self):
        dataset = ImageSampler(imgs=self.imgs, transform=self.transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_max,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        progress = self.progress_bar(dataloader.__len__(), verbose=self.verbose)
        features = []
        for batch_imgs in dataloader:
            batch_imgs = batch_imgs.to(self.device, self.precision)
            with torch.autocast(
                device_type=self.device,
                dtype=self.precision,
                enabled=(self.precision != torch.float32),
            ):
                batch_features = self.encoder(batch_imgs)
                progress.update()
            features.append(batch_features.to("cpu", dtype=torch.float32).numpy())
        progress.clear()
        # Concatenate features
        features = np.concatenate(features, axis=0)
        print(features.shape)
        return features.shape[0], features

    def save_features(self, dst, save_as="h5"):
        images, paths = [], []
        for i in range(self.nb_features):
            img, path = self.get_image(i, numpy=True)
            images.append(img)
            paths.append(path)
        images = np.stack(images, axis=0)
        paths = np.array(paths, dtype=h5py.string_dtype(encoding="utf-8"))
        # Save the features to disk
        os.makedirs(self.dst, exist_ok=True)
        features_path = os.path.join(dst, f"features_{self.encoder.enc_name}.{save_as}")
        if save_as == "h5":
            assets = {"features": self.features, "images": images}
            attributes = {
                "features": {
                    "encoder": self.encoder.enc_name,
                },
                "images": {"paths": paths},
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


class ImageSampler(Dataset):
    """Dataset from a WSI patcher to read tiles"""

    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        if not isinstance(img, np.ndarray):
            img = PIL.Image.open(img)
        else:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.dtype not in [np.uint8, np.float32]:
                img = np.float32(img)
            img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img


# class TileEncoder
# enable TileEncoding without custom coords_path
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
            assert self.name == self.slide.name, "tiles are from another slide"
        self.nb_features, self.features = self.extract_patch_features()
        if self.dst:
            self.feat_path = self.save_features(self.dst, self.save_as)

    def get_patch_attributes(self):
        try:
            self.mag_0 = self.tile_attr.get("magnification", None)
            self.pixel_size_0 = self.tile_attr.get("mpp", None)
            self.mag_target = self.tile_attr.get("target_magnification", None)
            self.pixel_size_target = self.tile_attr.get("target_mpp", None)
            self.patch_size_target = self.tile_attr.get("target_patch_size", None)
            self.overlap_target = self.tile_attr.get("target_overlap", None)
            self.level = self.tile_attr.get("level", None)
            self.level_size = self.tile_attr.get("level_size", None)
            self.patch_size_level = self.tile_attr.get("level_patch_size", None)
            self.overlap_level = self.tile_attr.get("level_overlap", None)
            self.tissu_thr = self.tile_attr.get("tissu_thr", None)
            self.name = self.tile_attr.get("name", None)
            if None in (
                self.patch_size_target,
                self.mag_0,
                self.mag_target,
                self.overlap_target,
            ):
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
            patch_size=self.patch_size_target,
            overlap=self.overlap_target,
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
            batch_tiles = batch_tiles.to(self.device, self.precision)
            with torch.autocast(
                device_type=self.device,
                dtype=self.precision,
                enabled=(self.precision != torch.float32),
            ):
                batch_features = self.encoder(batch_tiles)
                progress.update()
            features.append(batch_features.to("cpu", dtype=torch.float32).numpy())
        progress.clear()

        # Concatenate features
        features = np.concatenate(features, axis=0)
        return features.shape[0], features

    def leiden_patch_features(
        self,
        pcs: int | None = None,
        neighbors: int = 50,
        resolution: float = 0.3,
        solver: str = "arpack",
    ):
        features_clustered = ad.AnnData(
            X=self.features,
            obs=pd.DataFrame(self.tile_coords, columns=["x", "y", "w", "h"]),
        )
        pcs = min(pcs, *(self.features.shape))
        if pcs > 0:
            sc.pp.pca(features_clustered, n_comps=pcs, svd_solver=solver)

        neighbors = min(neighbors, self.features.shape[0] - 1)
        sc.pp.neighbors(
            features_clustered,
            n_neighbors=neighbors,
            n_pcs=pcs,
        )
        sc.tl.umap(features_clustered)  # Compute UMAP
        sc.tl.leiden(features_clustered, resolution=resolution)
        features_clustered.obs["leiden_int"] = features_clustered.obs["leiden"].astype(
            {"leiden": "int32"}
        )
        return features_clustered

    def rgb_patch_features(self, pcs: int = 3, solver: str = "arpack"):
        features_rgb = ad.AnnData(
            X=self.features,
            obs=pd.DataFrame(self.tile_coords, columns=["x", "y", "w", "h"]),
        )
        pcs = min(pcs, *(self.features.shape))
        if pcs < 3:
            solver = "full"
        sc.pp.pca(features_rgb, n_comps=pcs, svd_solver=solver)

        scaler = MinMaxScaler()
        scaled_features_rgb = scaler.fit_transform(features_rgb.obsm["X_pca"])
        features_rgb.obs[["r", "g", "b"]] = scaled_features_rgb
        return features_rgb

    def visualize_embeddings(
        self,
        method: str = "leiden",
        level_to_view: int = 4,
        alpha: float = 1,
        save_vis: str | None = None,
        show: bool = False,
        **kwargs,
    ) -> str:
        # Retrieve sampler function
        if method.lower() in ["leiden", "rgb"]:
            visualization = partial(
                getattr(self, method + "_patch_features"),
            )
        emb_data = visualization(**kwargs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
        if method == "leiden":
            visualise_tile_feat(
                self.slide,
                data=emb_data.obs,
                feat="leiden_int",
                analyse_level=self.level,
                level_to_view=level_to_view,
                cmap=plt.get_cmap("gist_rainbow"),
                alpha=alpha,
                ax=ax,
                title=f"{self.encoder.enc_name} {method}",
                loc=None,
                show=False,
            )
        elif method == "rgb":
            visualise_tile_rgb(
                self.slide,
                data=emb_data.obs,
                analyse_level=self.level,
                level_to_view=level_to_view,
                alpha=alpha,
                ax=ax,
                title=f"{self.encoder.enc_name} {method}",
                show=False,
            )
        else:
            raise ValueError("embeddings visualization method unknown")

        if show:
            plt.show
        # Save visualization
        if save_vis:
            os.makedirs(save_vis, exist_ok=True)
            vis_path = os.path.join(save_vis, f"{self.slide.name}_{method}.jpg")
            fig.savefig(vis_path)
        return vis_path

    def save_features(self, dst, save_as="h5"):
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


from slide.slide_encoder.load import SLIDE_TO_TILE_ENCODER_MATCH


@torch.inference_mode()
def aggragate_tiles_features(
    features_path: str,
    slide_encoder: torch.nn.Module,
    device: Optional[str] = "cuda",
    dst: Optional[str] = None,
    save_as: Optional[str] = "h5",
) -> str:

    # Set the slide encoder model to device and eval
    slide_encoder.to(device)
    slide_encoder.eval()

    xywh_attrs, xywh = read_h5_coords(features_path)
    tile_attrs, tile_feats = read_h5_features(features_path)
    slide_name = tile_attrs["name"]
    tile_encoder = tile_attrs["encoder"]
    # tile encoder sanity check:
    if not slide_encoder.enc_name.startswith("mean-"):
        try:
            SLIDE_TO_TILE_ENCODER_MATCH[slide_encoder.enc_name] == tile_encoder
        except ValueError as e:
            print(
                f"Tile features were extracted with a tile encoder which does not match the slide encoder provided"
            )

    # Convert slide_features to tensor
    tile_features = torch.from_numpy(tile_feats).float().to(device)
    tile_features = tile_features.unsqueeze(0)  # Add batch dimension

    coords = torch.from_numpy(xywh[:, :2]).to(device)
    coords = coords.unsqueeze(0)  # Add batch dimension

    # Prepare input batch dictionary
    batch = {"features": tile_features, "coords": coords, "attributes": xywh_attrs}

    # Generate slide-level features
    with torch.autocast(
        device_type=device,
        enabled=(slide_encoder.precision != torch.float32),
    ):
        slide_feats = slide_encoder(batch, device)
    slide_feats = slide_feats.float().cpu().numpy().squeeze()

    # Save slide-level features if save path is provided
    if dst:
        os.makedirs(dst, exist_ok=True)
        save_path = os.path.join(dst, f"{slide_name}.{save_as}")
        assets = {
            "features": slide_feats,
            "tile_features": tile_feats,
            "coords": xywh,
        }
        attributes = {
            "features": {
                "encoder": slide_encoder.enc_name,
                "name": slide_name,
                "dst": dst,
            },
            "tile_features": tile_attrs,
            "coords": xywh_attrs,
        }
        save_h5(
            save_path,
            assets=assets,
            attributes=attributes,
            mode="w",
        )
    else:
        raise ValueError(f"Invalid save_as: {save_as}. Only h5 is supported.")

    return save_path


# add possibility to generate random biopsies
class EncodingSampler(Dataset):
    """
    Class for EncodingSampler, which samples the WSI embeddings according to a sampling strategy.
    """

    # Defining valid options for samplers
    valid_samplers = ["all", "random", "random_strict", "niche"]
    no_n_samples = ["all"]
    no_total_tiles = ["niche"]
    with_tile_coords = ["niche"]

    def __init__(
        self,
        sampler_name: str,
        feat_path: str,
        n_samples: int = None,
        q_samples: float = None,
        min_samples: int = 3,
        hop: int = 2,
        read_feat: bool = True,
        read_coords: bool = True,
        total_tiles: int = None,
        coords: np.ndarray = None,
        patch_dim: tuple[int] = None,
    ):
        """
        Build an instance of EncodingSampler, which samples tiles from the WSI embeddings according to the given
        sampler name.

        Args:
            sampler_name:       str, name of the sampler to be used.
            feat_path:          str, path to the WSI embedding file.
            n_samples:          int, number of tiles to sample, default=None.
            q_samples:          int, percentage of tiles to sample, default=None.
            min_samples:        int, minimum number of tiles to sample, default=3.
            hop:                int, maximum hop neighbors to look at next to the niche center, default=2.
            read_file:          bool, whether to use the read the WSI embedding files in the sampler or not,
                                default=True, meaning that the information are retrieved elsewhere and given to the
                                function.
            total_tiles:        int, total number of tiles in the WSI, default=None.
            coords:             np.ndarray, the coordinates of the tiles, default=None.
            patch_dim:          tuple, the patch dimension (width, height), default=None.

        """
        # Storing useful variables
        self.sampler_name = sampler_name
        self.feat_path = feat_path
        self.name, _ = os.path.splitext(os.path.basename(feat_path))
        self.n_samples = n_samples
        self.q_samples = q_samples
        self.hop = hop
        self.read_feat = read_feat
        self.read_coords = read_coords
        self.total_tiles = total_tiles
        self.coords = coords
        self.patch_dim = patch_dim
        self.min_samples = min_samples

        # Check that a number of samples was given
        if sampler_name not in EncodingSampler.no_n_samples:
            assert (n_samples is not None and n_samples > 0) or (
                q_samples is not None and 0 < q_samples <= 1
            ), (
                f"Please provide a valid value for n_samples (int > 0) or q_samples (0 < float <= 1) "
                + f"when sampler name is not in: {', '.join(EncodingSampler.no_n_samples)}."
            )

        # Reading file information
        if self.read_feat:
            # Retrieve features and attributes and coordinates
            _, self.features = self.read_h5_feat(feat_path)
            self.total_tiles = self.features.shape[0]
            if self.q_samples is not None:
                self.n_samples = max(
                    int(self.q_samples * self.total_tiles), self.min_samples
                )

        if self.read_coords:
            self.attributes, self.coords = self.read_h5_coord(feat_path)
            # Retrieve patch's dimensions
            self.patch_dim = (
                self.attributes["level_patch_size"],
                self.attributes["level_patch_size"],
            )

        # Retrieve sampler function
        if sampler_name.lower() in self.valid_samplers:
            self.sampler = partial(
                getattr(self, self.sampler_name + "_sampler"),
                **{
                    "n_samples": self.n_samples,
                    "total_tiles": self.total_tiles,
                    "hop": self.hop,
                },
            )
        else:
            raise ValueError(
                f"Invalid sampler: {sampler_name}."
                + f"\nPlease choose a valid sampler from : {', '.join(self.valid_samplers)}"
            )

        self.sample_ids, self.sample_feats = self.get_feat()

    def __len__(self):
        """
        Get the total number of tiles.
        Returns:
            n_tiles:    int, total number of tiles.
        """
        return self.total_tiles

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.sample_feats[idx]

    def get_feat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tiles and get their corresponding features.

        Returns:
            sample_id:      torch.tensor, size (n_samples,), selected tiles' ID.
            feats:          torch.tensor, size (n_samples, dim_feats), features corresponding to the selected tiles.

        """
        if (
            self.sampler_name not in EncodingSampler.no_n_samples
            and self.n_samples is None
            and self.q_samples is not None
        ):
            self.n_samples = max(
                int(self.total_tiles * self.q_samples), self.min_samples
            )
        samples_id = self.sampler(n_samples=self.n_samples)
        if self.read_feat:
            feat = self.features[samples_id]
        else:
            feat = self.read_h5_feat(self.feat_path)[samples_id]
        return samples_id, feat

    def random_sampler(
        self, n_samples: int, total_tiles: int = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Randomly sample a number of tiles, and return their corresponding indices.
        Args:
            n_samples:      int, number of tiles to select.
            total_tiles:    int, total number of tiles in the WSI, default=None.


        Returns:
            indices:        torch.tensor, size (n_samples,), selected tiles' ID.

        """

        # Default args parsing
        if total_tiles is None:
            total_tiles = self.total_tiles
        assert total_tiles is not None and total_tiles > 0, (
            "Invalid total_tiles value: {}".format(total_tiles)
            + "\nPlease provide a valid value (int > 0) for total_tiles."
        )

        # Check min samples
        n_samples = max(n_samples, self.min_samples)

        # Returning selected indices
        return torch.randint(0, total_tiles, (n_samples,))

    def all_sampler(self, total_tiles: int = None, *args, **kwargs) -> torch.Tensor:
        """
        Return all tiles' indices.

        Args:
            total_tiles:    int, total number of tiles in the WSI, default=None.

        Returns:
            indices:        torch.tensor, size (n_samples,), all tiles' ID.

        """

        # Default args parsing
        if total_tiles is None:
            total_tiles = self.total_tiles
        assert total_tiles is not None and total_tiles > 0, (
            "Invalid total_tiles value: {}".format(total_tiles)
            + "\nPlease provide a valid value (int > 0) for total_tiles."
        )

        # Returning the selected indices
        return torch.IntTensor(list(range(total_tiles)))

    def random_strict_sampler(
        self, n_samples: int, total_tiles: int = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Randomly sample a number of tiles and return their corresponding indices, only if the bag is bigger than
        n_samples.

        Args:
            n_samples:      int, number of tiles to select.
            total_tiles:    int, total number of tiles in the WSI, default=None.

        Returns:
            indices:        torch.tensor, size (n_samples,), selected tiles' ID.

        """
        # Default args parsing
        if total_tiles is None:
            total_tiles = self.total_tiles
        assert total_tiles is not None and total_tiles > 0, (
            "Invalid total_tiles value: {}".format(total_tiles)
            + "\nPlease provide a valid value (int > 0) for total_tiles."
        )

        # Returning the indices selected with the corresponding method
        if total_tiles >= n_samples:
            return self.random_sampler(n_samples, total_tiles=total_tiles)
        else:
            return self.all_sampler()

    def niche_sampler(
        self,
        n_samples: int,
        hop: int = None,
        coords: np.ndarray = None,
        patch_dim: tuple[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Randomly sample tiles within different niches, where the maximum hop neighborhood is determined by hop
        Return the selected tiles' corresponding indices.

        Args:
            n_samples:      int, number of tiles to select.
            hop:            int, maximum hop neighbors to look at next to the niche center, default=None.
            coords:         np.ndarray, size (n_samples, 2), tiles center coordinates, default=None, meaning that it
                            will be retrieved from the .h5 file.
            patch_dim:      tuple, size (height, width), patch dimensions, default=None, meaning that it will be
                            retrieved from the .h5 file.

        Returns:
            indices:        torch.tensor, size (n_samples,), selected tiles' ID.

        """

        # Parse default args
        if hop is None:
            hop = self.hop
        if coords is None:
            coords = self.coords
        if patch_dim is None:
            patch_dim = self.patch_dim

        # Checking minimum samples
        n_samples = max(n_samples, self.min_samples)

        # Check that all the information are given
        assert patch_dim is not None, (
            "Invalid patch_dim value: {}.".format(patch_dim)
            + "\n Please provide a tuple of ints (width, height) in number of pixels for patch_dim."
        )
        assert coords is not None and isinstance(coords, np.ndarray), (
            "Invalid coords value: {}.".format(coords)
            + "\nPlease provide an array of size (total_tiles, 2)."
        )

        # Initialize indices, number of niches and niche centers
        indices = []
        n_niches = int(n_samples / (2 * hop + 1) ** 2)
        seeds = self.random_sampler(n_niches)

        # For each niche center, retrieve the tils in the niche
        for i, s in enumerate(seeds):
            seed_x, seed_y = coords[s, :2]
            neigh_x = np.arange(
                seed_x - hop * patch_dim[0],
                seed_x + hop * patch_dim[0] + 1,
                patch_dim[0],
            )
            neigh_y = np.arange(
                seed_y - hop * patch_dim[1],
                seed_y + hop * patch_dim[1] + 1,
                patch_dim[1],
            )
            inx = np.nonzero(np.isin(coords[:, 0], neigh_x))[0]
            iny = np.nonzero(np.isin(coords[:, 1], neigh_y))[0]
            inter = np.intersect1d(inx, iny)
            indices += inter.tolist()

        # Returning the list of selected tiles' indices
        return torch.IntTensor(list(set(indices)))

    def read_h5_feat(self, path: str) -> tuple[dict, np.ndarray]:
        """
        Get embeddings and attributes from a WSI embedding file.

        Args:
            path:       str, path to the WSI embedding file.

        Returns:
            attrs:      dict, dictionary containing the corresponding WSI attributes.
            feats:      np.ndarray, size (n_tiles, dim_feats), array containing the WSI embeddings.

        """

        # Retrieving the h5 features and attributes
        return read_h5_features(embs_path=path)

    def read_h5_coord(self, path: str) -> tuple[dict, np.ndarray]:
        """
        Get coordinates and attributes from a WSI embedding file.

        Args:
            path:       str, path to the WSI embedding file.

        Returns:
            attrs:      dict, dictionary containing the corresponding WSI attributes.
            coords:     np.ndarray, size (n_tiles, 2), array containing the WSI coordinates.

        """
        return read_h5_coords(coords_path=path)
