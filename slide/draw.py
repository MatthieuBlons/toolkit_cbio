# General libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, patches

from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
import cv2
from PIL import Image
import openslide
import pandas as pd

# Project modules
import osfile.manager as fm
from slide.utils import get_x_y_to, get_size_to, get_slide_reader, get_slide_whole
from draw.utils import add_annotation_box
from slide.reader import OpenWSI
from bioformat.reader import OpenOME


def add_slide_annotation_box(
    img: np.ndarray, slide_attrs: dict, title: str = "", *args, **kwargs
):
    """
    Add slide annotations in a box on the image.

    Args:
        img:                np.ndarray, image onto which to add slide annotations.
        slide_attrs:        dict, dictionary containing the slide information to add as annotations.
        title:              str, title of to add to the annotation box in the image, default=''.
        *args:              list or tuple, additional positional arguments to pass to the annotation box.
        **kwargs:           dict, additional keyword arguments to pass to the annotation box.

    Returns:
        img:                np.ndarray, image onto which the slide annotations were added.

    """

    # Creating text annotation list
    text = []
    for key, value in slide_attrs.items():
        if key.lower() == "size":
            if not isinstance(value, tuple):
                value = (value, value)
            text.append("width={} px, height={} px".format(*value))
        elif key.lower() == "mpp":
            text.append("mpp={:.4f}".format(value))
            if "mag_0" in slide_attrs.keys():
                text[-1] += ", {}x".format(slide_attrs["mag_0"])
        elif key.lower() == "downsample":
            text.append("downsample={}".format(value))
            if "mag_0" in slide_attrs.keys():
                text[-1] += " from {}x".format(slide_attrs["mag_0"])
        elif key.lower() == "strategy":
            text.append("masking strategy={}".format(value))
        elif key.lower() == "tissue_tolerance":
            text.append("tissue tolerance={:.1f}".format(value * 100) + "%")
        elif key.lower() == "patch_size_target":
            text.append("ask: patch={}".format(value))
            if "overlap_target" in slide_attrs.keys():
                text[-1] += " w. overlap={}".format(slide_attrs["overlap_target"])
            if "mag_target" in slide_attrs.keys():
                text[-1] += " at {}x".format(slide_attrs["mag_target"])
        elif key.lower() == "patch_size_mask":
            text.append("out: patch={}".format(value))
            if "overlap_level" in slide_attrs.keys():
                text[-1] += " w. overlap={}".format(slide_attrs["overlap_level"])
            if "level" in slide_attrs.keys():
                text[-1] += " at lvl={}".format(slide_attrs["level"])
        elif key.lower() not in [
            "mag_0",
            "mag_target",
            "overlap_target",
            "overlap_level",
            "level",
        ]:
            text.append("{}={}".format(key.replace("_", " "), value))

    # Adding annotations in a box on the image
    img = add_annotation_box(img=img, text=text, title=title, *args, **kwargs)

    # Returning the annotated image
    return img


def map_cell_feat(
    slide: str | OpenWSI | OpenOME | openslide.OpenSlide,
    data: pd.DataFrame,
    feat: str,
    analyse_level: int = 0,
    level_to_view: int = 0,
    discrete: bool = True,
    color: dict | list | None = None,
    cmap: plt.cm = plt.cm.jet,
    alpha: float = 0.6,
    limits: tuple[float, float] = (0, 1),
    ax: axes.Axes | None = None,
    title: str | None = None,
    loc: str = "right",
    show: bool = True,
):
    """
    Overlay cell-level feature on the image.

    Args:
        slide:              openslide.OpenSlide like, slide to process.
        data:               pandas.DataFrame each rows is a tile with its coords (x, y, w, h) and feat in columns.
        feat:               key feature to be desplayed on the image.
        analyse_level:      pyramid level at which data was extracted.
        level_to_view:      pyramid level to be displayed.
        discrete:           is feat a discrete or continuous.
        color:              dictionary of color to use if descrete == True.
        cmap:               color map to use if feat is continuous.
        alpha:              transparancy to apply.
        limits:             feat (min, max) values to display.
        ax:                 axes for plot.
        title:              axes title.
        loc:                location to display legend.
        show:               show plot.

    Returns:
        None

    """
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    ax.set_title(title, size=20)
    ax.axis("off")

    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if level_to_view > slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]
    dim_at_level_to_view = slide.level_dimensions[level_to_view]
    down_at_level_to_view = slide.level_downsamples[level_to_view]

    wsi = slide.read_region(
        (0, 0), level_to_view, slide.level_dimensions[level_to_view]
    )
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    ax.imshow(wsi, aspect="equal")
    if discrete:
        # color code the different labels
        values = data[feat].unique().tolist()
        values = [value for value in values if ~np.isnan(value)]
        if isinstance(color, list):
            colors = color
            assert not len(colors) != len(
                values
            ), "color list must match number of non nan values"
        else:
            hues = np.linspace(0, 1, len(values), endpoint=False)  # Evenly spaced hues
            colors = cmap(hues)
        # add cells
        for i, value in enumerate(values):
            for _, row in data.loc[data[feat] == value].iterrows():
                x, y = get_x_y_to(
                    (row["x"], row["y"]),
                    dim_at_analyse_level,
                    dim_at_level_to_view,
                    integer=True,
                )
                w, h = get_size_to(
                    (row["w"], row["h"]),
                    down_at_analyse_level,
                    down_at_level_to_view,
                    integer=True,
                )
                plot_seed = (x, y)
                patch = patches.Rectangle(
                    plot_seed,
                    w,
                    h,
                    fill=True,
                    facecolor=colors[i],
                    alpha=alpha,
                )
                ax.add_patch(patch)
        # add legend
        ax.legend(
            handles=[
                patches.Patch(color=colors[i], label=label)
                for i, label in enumerate(values)
            ],
            loc=loc,
            borderaxespad=1,
            frameon=True,
            facecolor="white",
            framealpha=0.7,
        )
    else:
        norm = plt.Normalize(limits[0], limits[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for _, row in data.iterrows():
            x, y = get_x_y_to(
                (row["x"], row["y"]),
                dim_at_analyse_level,
                dim_at_level_to_view,
                integer=True,
            )
            w, h = get_size_to(
                (row["w"], row["h"]),
                down_at_analyse_level,
                down_at_level_to_view,
                integer=True,
            )
            plot_seed = (x, y)
            value = row[feat]
            colors = cmap(norm(value))
            patch = patches.Rectangle(
                plot_seed,
                w,
                h,
                fill=True,
                facecolor=colors,
                alpha=alpha,
            )
            ax.add_patch(patch)
        plt.colorbar(sm, ax=ax, label=feat, location=loc)

    if show:
        plt.show()


def visualise_tile_feat(
    slide: str | OpenWSI | OpenOME | openslide.OpenSlide,
    data: pd.DataFrame,
    feat: str,
    region=(0, 0),
    size=None,
    analyse_level: int = 0,
    level_to_view: int = 0,
    cmap: plt.cm = plt.cm.jet,
    alpha: float = 0.6,
    smooth: int | None = None,
    limits: tuple[float, float] = (0, 1),
    ax: axes.Axes | None = None,
    legend: list | None= None,
    title: str | None = None,
    loc: str = "right",
    show: bool = True,
):
    """
    Overlay cell-level feature on the image.

    Args:
        slide:              openslide.OpenSlide like, slide to process.
        data:               pandas.DataFrame each rows is a tile with its coords (x, y, w, h) and feat in columns.
        feat:               key feature to be desplayed on the image.
        analyse_level:      pyramid level at which data was extracted.
        level_to_view:      pyramid level to be displayed.
        cmap:               color map to use if feat is continuous.
        alpha:              transparancy to apply.
        smooth:             sigma for gaussian_filter.
        limits:             feat (min, max) values to display.
        ax:                 axes for plot.
        title:              axes title.
        loc:                location to display legend.
        show:               show plot.

    Returns:
        None
    """
    # here numpy can be false and I get the dim with (pyvips.image.get("width"), pyvips.image.get("height"))
    # should also allow pyvips images as slide
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    ax.set_title(title, size=20)
    ax.axis("off")

    # read slide if path is provided
    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    # check downsampling level
    if level_to_view > slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    down_analyse = slide.level_downsamples[analyse_level]
    down_view = slide.level_downsamples[level_to_view]

    if not size:
        size = slide.level_dimensions[analyse_level]
        dim_view = slide.level_dimensions[level_to_view]
    else:
        dim_view = (
            int(size[0] * (down_analyse / down_view)),
            int(size[1] * (down_analyse / down_view)),
        )

    # get whole image
    wsi = slide.read_region(location=region, level=level_to_view, size=dim_view)
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    # normalize features
    values = data[feat].values
    vmin, vmax = limits
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Vectorized coordinates and sizes
    xs = data["x"].values
    ys = data["y"].values
    ws = data["w"].values
    hs = data["h"].values

    # Filter tiles that intersect the selected region
    x0, y0 = region
    W, H = size

    mask = (xs + ws >= x0) & (xs < x0 + W) & (ys + hs >= y0) & (ys < y0 + H)

    xs, ys, ws, hs, values = xs[mask], ys[mask], ws[mask], hs[mask], values[mask]

    # Shift to region coordinates
    xs = xs - x0
    ys = ys - y0

    # or apply get_size_to() get_x_y_to()
    x_scaled = (xs * dim_view[0] / size[0]).astype(int)
    y_scaled = (ys * dim_view[1] / size[1]).astype(int)
    w_scaled = np.maximum((ws * down_analyse / down_view).astype(int), 1)
    h_scaled = np.maximum((hs * down_analyse / down_view).astype(int), 1)

    # Create transparent overlay
    overlay = np.zeros((dim_view[1], dim_view[0], 4), dtype=np.float32)  # HWC RGBA
    for x, y, w, h, val in zip(x_scaled, y_scaled, w_scaled, h_scaled, values):
        color = cmap(norm(val), alpha=alpha)  # RGBA in [0,1]
        overlay[y : y + h + 1, x : x + w + 1, :] = color

    # gaussian smoothing
    if smooth is not None:
        overlay = gaussian_filter(overlay, sigma=(smooth, smooth, 0))

    ax.imshow(wsi, aspect="equal")
    ax.imshow(overlay, aspect="equal")
    ax.set_title(title, size=16)
    ax.axis("off")

    if loc:
        plt.colorbar(sm, ax=ax, label=feat, location=loc)
    if show:
        plt.show()


def visualise_tile_rgb(
    slide: str | OpenWSI | OpenOME | openslide.OpenSlide,
    data: pd.DataFrame,
    region=(0, 0),
    size=None,
    analyse_level: int = 0,
    level_to_view: int = 0,
    alpha: float = 0.6,
    smooth: int | None = None,
    ax: axes.Axes | None = None,
    title: str | None = None,
    show: bool = True,
):
    """
    Overlay cell-level feature on the image.

    Args:
        slide:              openslide.OpenSlide like, slide to process.
        data:               pandas.DataFrame each rows is a tile with its coords (x, y, w, h) and (r, g, b) E [0, 1] values in 3 distincts columns.
        analyse_level:      pyramid level at which data was extracted.
        level_to_view:      pyramid level to be displayed.
        alpha:              transparancy to apply.
        smooth:             sigma for gaussian_filter.
        ax:                 axes for plot.
        title:              axes title.
        loc:                location to display legend.
        show:               show plot.

    Returns:
        None
    """
    # get rgb image
    assert all([k in data.keys() for k in ["r", "g", "b"]]), print(
        "data must containt r, g, b features"
    )
    rgbs = data[["r", "g", "b"]].to_numpy()

    # here numpy can be false and I get the dim with (pyvips.image.get("width"), pyvips.image.get("height"))
    # should also allow pyvips images as slide
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    ax.set_title(title, size=20)
    ax.axis("off")

    # read slide if path is provided
    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    # check downsampling level
    if level_to_view > slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    down_analyse = slide.level_downsamples[analyse_level]
    down_view = slide.level_downsamples[level_to_view]

    if not size:
        size = slide.level_dimensions[analyse_level]
        dim_view = slide.level_dimensions[level_to_view]
    else:
        dim_view = (
            int(size[0] * (down_analyse / down_view)),
            int(size[1] * (down_analyse / down_view)),
        )

    # get whole image
    wsi = slide.read_region(location=region, level=level_to_view, size=dim_view)
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    # Vectorized coordinates and sizes
    xs = data["x"].values
    ys = data["y"].values
    ws = data["w"].values
    hs = data["h"].values

    # Filter tiles that intersect the selected region
    x0, y0 = region
    W, H = size

    mask = (xs + ws >= x0) & (xs < x0 + W) & (ys + hs >= y0) & (ys < y0 + H)

    xs, ys, ws, hs, values = xs[mask], ys[mask], ws[mask], hs[mask], rgbs[mask]

    # Shift to region coordinates
    xs = xs - x0
    ys = ys - y0

    # or apply get_size_to() get_x_y_to()
    x_scaled = (xs * dim_view[0] / size[0]).astype(int)
    y_scaled = (ys * dim_view[1] / size[1]).astype(int)
    w_scaled = np.maximum((ws * down_analyse / down_view).astype(int), 1)
    h_scaled = np.maximum((hs * down_analyse / down_view).astype(int), 1)

    # Create transparent overlay
    overlay = np.zeros((dim_view[1], dim_view[0], 4), dtype=np.float32)  # HWC RGBA
    for x, y, w, h, rgb in zip(x_scaled, y_scaled, w_scaled, h_scaled, values):
        rgba = np.concatenate([rgb, np.ones(1) * alpha], axis=0)
        overlay[y : y + h + 1, x : x + w + 1, :] = np.concatenate(
            [rgb, np.ones(1) * alpha], axis=0
        )

    # gaussian smoothing
    if smooth is not None:
        overlay = gaussian_filter(overlay, sigma=(smooth, smooth, 0))

    ax.imshow(wsi, aspect="equal")
    ax.imshow(overlay, aspect="equal")
    ax.set_title(title, size=16)
    ax.axis("off")

    if show:
        plt.show()


def draw_cut(
    slide: str | OpenWSI | OpenOME | openslide.OpenSlide,
    xywh: list | np.ndarray,
    analyse_level: int = 0,
    level_to_view: int = 0,
    color: str = "red",
    title: str | None = None,
    ax: axes.Axes | None = None,
    show: bool = True,
):
    """
    Plots the patches you are going to extract from the slide. So that they
    appear as red boxes on the lower resolution of the slide.

    Args:
        slide:              str or openslide like object.
        xywh:               list of coords to extract tiles from slide.
        analyse_level:      level at which coords were extrqcted.
        level_to_view:      level use for display.
        color:              color of the box.
        title:              axes title.
        ax:                 axes if provided.
        show:               display figure.

    Return:
        None
    """
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")

    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if level_to_view > slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]

    dim_at_level_to_view = slide.level_dimensions[level_to_view]
    down_at_level_to_view = slide.level_downsamples[level_to_view]

    wsi = slide.read_region(
        (0, 0), level_to_view, slide.level_dimensions[level_to_view]
    )
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    ax.imshow(wsi, aspect="equal")
    for para in xywh:
        x, y, w, h = para
        x, y = get_x_y_to(
            (x, y), dim_at_analyse_level, dim_at_level_to_view, integer=True
        )
        w, h = get_size_to(
            (w, h), down_at_analyse_level, down_at_level_to_view, integer=True
        )
        plot_seed = (x, y)
        patch = patches.Rectangle(plot_seed, w, h, fill=False, edgecolor=color)
        ax.add_patch(patch)
    ax.set_title(title, size=20)
    ax.axis("off")

    if show:
        plt.show()


def visualize_cut(
    wsi_slide: str | OpenWSI | OpenOME | openslide.OpenSlide,
    coords: np.ndarray | list[list | tuple],
    size: tuple[int, int],
    patch_size_target: int,
    mag_target: int,
    slide_attrs: dict = None,
    save_cut: str = None,
    show_plot: bool = False,
    annotation: bool = True,
    *args,
    **kwargs,
) -> str:
    """
    Visualize the patches generated from the tissue segmentation mask and the grid coordinate computed at the target
    resolution.

    Args:
        wsi_slide:          str or OpenWSI, OpenWSI object representing the slide or the path to the slide file to
                            load in a OpenWSI object.
        coords:             np.ndarray or list of list or tuple, array of size (n_tiles, 5) containing the tiles
                            coordinates (x,y,w,h).
        size:               tuple of int, the desired size of the image on which to plot the patches.
        patch_size_target:  int, the desired size of the patches in the target resolution.
        mag_target:         int, the desired magnification of the patches in the target resolution.
        slide_attrs:        dict, the slide attributes to be added as annotations on the image in the annotation box,
                            default=None, meaning there is no additional information.
        save_cut:           str, the path to the directory where to save the visualization of the patches,
                            default=None.
        show_plot:          bool, whether to show the visualization of the patches, default=False.

    Returns:
        cut_path:       str, the path to the visualization of the patches including the file name.

    """

    # Parsing default slide attrs
    if slide_attrs is None:
        slide_attrs = {}

    # Retrieve slide if not given but only path
    if isinstance(wsi_slide, str):
        reader_ = get_slide_reader(wsi_slide)
        wsi_slide = reader_(wsi_slide)
    else:
        wsi_slide = wsi_slide

    # Computing best level, patch size, overlap in original dimensions
    downsample = wsi_slide.magnification / mag_target
    level, downsample_level, resize_factor = wsi_slide.get_best_level_for_downsample(
        downsample
    )
    patch_size_level = round(patch_size_target / resize_factor)

    # Get thumbnail image of the slide and compute downsample factor and corresponding patch size
    thumbnail = np.array(wsi_slide.get_thumbnail(size))[:, :, :3]
    thumbnail_height, thumbnail_width, _ = thumbnail.shape
    downsample_factor = max(
        wsi_slide.level_dimensions[level][0] / thumbnail_width,
        wsi_slide.level_dimensions[level][1] / thumbnail_height,
    )
    thumbnail_patch_size = max(1, int(patch_size_level / downsample_factor))

    # Draw rectangles for patches
    for x_, y_, _, _ in coords:
        x_, y_ = get_x_y_to(
            point=(x_, y_),
            dim_from=wsi_slide.level_dimensions[level],
            dim_to=(thumbnail_width, thumbnail_height),
            integer=True,
        )
        thickness = max(1, thumbnail_patch_size // 16)
        thumbnail = cv2.rectangle(
            img=thumbnail,
            pt1=(x_, y_),
            pt2=(x_ + thumbnail_patch_size, y_ + thumbnail_patch_size),
            color=(255, 0, 0),
            thickness=thickness,
        )

    # Adding annotations
    all_slide_attrs = {
        "size": wsi_slide.dimensions,
        "mpp": wsi_slide.mpp,
        "mag_0": wsi_slide.magnification,
        "patch_size_target": patch_size_target,
        "mag_target": mag_target,
        "patch_size_mask": patch_size_level,
    }
    all_slide_attrs.update(slide_attrs)
    if "text_box_height" not in kwargs.keys():
        kwargs["text_box_height"] = 180
    if "text_box_color" not in kwargs.keys():
        kwargs["text_box_color"] = (204, 139, 189)
    if annotation:
        thumbnail = add_slide_annotation_box(
            thumbnail,
            slide_attrs=all_slide_attrs,
            title=f"{len(coords)} patches",
            *args,
            **kwargs,
        )

    # Showing the figure
    if show_plot:
        _, axis = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
        axis.set_axis_off()
        axis.imshow(thumbnail, aspect="equal")
        plt.show()

    # Saving the visualization
    if save_cut is not None:
        os.makedirs(save_cut, exist_ok=True)
        cut_path = os.path.join(save_cut, f"{wsi_slide.name}.jpg")
        Image.fromarray(thumbnail).save(cut_path)
    else:
        cut_path = None

    # Returning path to saved image
    return cut_path


def visualize_tissue_seg(
    mask: np.ndarray,
    size: tuple[int, int],
    slide_name: str,
    slide_attrs: dict = None,
    annotation: bool = True,
    save_seg: str = None,
    show: bool = False,
    *args,
    **kwargs,
) -> str:
    """
    Visualize the tissue segmentation mask.

    Args:
        mask:               np.ndarray, array containing the tissue segmentation mask.
        size:               tuple of int, size of the tissue segmentation mask image.
        slide_name:         str, name of the slide, which will be used to save the visualization image.
        slide_attrs:        dict, the slide attributes that will be added to the image in an annotation box, default=None.
        save_seg:           str, the path to the directory where to save the segmentation mask, default=None.
        show:               bool, whether to show the visualization of the tissue segmentation mask, default=False.

    Returns:
        seg_path:       str, the path to the segmentation mask including the file's name.
                        Returns None if the segmentation mask was not saved.
    """

    # Computing thumbnail size
    mask_height, mask_width = mask.shape
    if mask_width > mask_height:
        thumbnail_width = size[0]
        thumbnail_height = int(size[1] * mask_height / mask_width)
    else:
        thumbnail_height = size[1]
        thumbnail_width = int(size[0] * mask_width / mask_height)
    thumbnail_dim = (thumbnail_width, thumbnail_height)

    # Converting to grayscale and resizing mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask, thumbnail_dim, interpolation=cv2.INTER_LINEAR)

    # Adding annotations
    if annotation:
        mask = add_slide_annotation_box(
            mask, slide_attrs=slide_attrs, title="Tissue mask", *args, **kwargs
        )

    # Showing the tissue segmentation image
    if show:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
        ax.set_axis_off()
        ax.imshow(mask, aspect="equal")
        plt.show()

    # Saving the tissue segmentation mask visualization
    if save_seg is not None:
        os.makedirs(save_seg, exist_ok=True)
        seg_path = os.path.join(save_seg, f"{slide_name}.jpg")
        Image.fromarray(mask).save(seg_path)
    else:
        seg_path = None

    # Returning path to the saved image
    return seg_path


def make_openslide_thumbnail():
    return None


def make_feature_thumbnail():
    return None


def multiple_openslide_thumbnail(
    data_dir: str,
    outputs_dir: str = None,
    level: int = 0,
    extension: str = "ndpi",
    grayscale: bool = False,
    verbose: bool = True,
):
    """
    Make thumbnail visualization of all slides (full) stored in a directory.

    Args:
        data_dir:           directory where slides are stored.
        outputs_dir:        output dir where to store the thumbnails.
        level:              pyramidal level to use.
        extension:          slide extension to look for.
        grayscale:          transform to grayscale
        verbose:            verbosity.

    Returns:
        None
    """
    # Finding files corresponding to the format in data dir
    files, cnt = fm.findFile(dir=data_dir, strings=extension, fileExtensions=True)
    if verbose:
        print(f"{cnt} WSI.{extension} were found in source dir: {data_dir}")

    # Make outputs directory if it does not exist
    if outputs_dir is not None:
        outputs_dir = os.path.join(data_dir, "visualization")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)

    # Make thumbnail visualization for all files found in data dir
    # call make_openslide_thumbnail
    for file in files:

        # Retrieve filename and file taf
        filename, _ = os.path.splitext(file)
        tag = os.path.basename(filename)

        # Get corresponding WSI and convert to grayscale if necessary
        wsi_np = get_slide_whole(file, level=level)
        if grayscale:
            wsi_np = (rgb2gray(wsi_np).astype("float32") - 1) * -1

        # Creating thumbnail visualization and saving it in outputs dir
        wsi_aspect = wsi_np.shape[1] / wsi_np.shape[0]
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(12, 12 / wsi_aspect), layout="constrained"
        )
        ax.set_title(f"{tag} at level={level}")
        ax.imshow(wsi_np)
        ax.axis("off")
        fig.savefig(os.path.join(outputs_dir, f"{tag}_slide_level_{level}.png"))
        plt.close()
        if verbose:
            print(f"Slide thumbnail saved in {outputs_dir}.")
