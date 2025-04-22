import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from slide.utils import get_x_y_to, get_size_to, get_slide_reader


def map_cell_feat(
    slide,
    df,
    feat,
    analyse_level=0,
    res_to_view=0,
    discrete=True,
    plot_args=None,
    ax=None,
    show=True,
):
    if not plot_args:
        plot_args = {
            "colors": plt.get_cmap("rainbow"),
            "alpha": 1,
            "size": (10, 10),
            "title": "",
        }
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=plot_args["size"], layout="constrained")
    ax.set_title(plot_args["title"], size=20)
    ax.axis("off")

    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if res_to_view > slide.level_count:
        print(
            f"downsampling level={res_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        res_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]
    wsi = slide.read_region((0, 0), res_to_view, slide.level_dimensions[res_to_view])
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]
    dim_at_res_to_view = slide.level_dimensions[res_to_view]
    down_at_res_to_view = slide.level_downsamples[res_to_view]
    ax.imshow(wsi, aspect="equal")

    if discrete:
        # color code the different labels
        values = df[feat].unique().tolist()
        values = [value for value in values if ~np.isnan(value)]
        if isinstance(plot_args["colors"], list):
            colors = plot_args["colors"]
            assert not len(colors) != len(
                values
            ), "color list must match number of non nan values"
        else:
            hues = np.linspace(0, 1, len(values), endpoint=False)  # Evenly spaced hues
            colors = plot_args["colors"](hues)
        # add cells
        for i, value in enumerate(values):
            for _, row in df.loc[df[feat] == value].iterrows():
                x, y = get_x_y_to(
                    (row["x"], row["y"]),
                    dim_at_analyse_level,
                    dim_at_res_to_view,
                    integer=True,
                )

                w, h = get_size_to(
                    (row["w"], row["h"]),
                    down_at_analyse_level,
                    down_at_res_to_view,
                    integer=True,
                )
                plot_seed = (x, y)
                patch = patches.Rectangle(
                    plot_seed,
                    w,
                    h,
                    fill=True,
                    facecolor=colors[i],
                    alpha=plot_args["alpha"],
                )
                ax.add_patch(patch)
        # add legend
        ax.legend(
            handles=[
                patches.Patch(color=colors[i], label=label)
                for i, label in enumerate(values)
            ],
            loc="lower right",
            borderaxespad=1,
            frameon=True,
            facecolor="white",
            framealpha=0.7,
        )
    else:
        minv, maxv = df[feat].min(), df[feat].max()
        norm = plt.Normalize(minv, maxv)
        sm = plt.cm.ScalarMappable(cmap=plot_args["colors"], norm=norm)
        sm.set_array([])
        for _, row in df.iterrows():
            x, y = get_x_y_to(
                (row["x"], row["y"]),
                dim_at_analyse_level,
                dim_at_res_to_view,
                integer=True,
            )
            w, h = get_size_to(
                (row["w"], row["h"]),
                down_at_analyse_level,
                down_at_res_to_view,
                integer=True,
            )
            plot_seed = (x, y)
            value = row[feat]
            colors = plot_args["colors"](norm(value))
            patch = patches.Rectangle(
                plot_seed,
                w,
                h,
                fill=True,
                facecolor=colors,
                alpha=plot_args["alpha"],
            )
            ax.add_patch(patch)
        fig.colorbar(sm, ax=ax, label=feat, location="bottom")

    if show:
        plt.show()


def visualise_tile_feat(
    slide,
    tile_feat,
    name,
    analyse_level=0,
    res_to_view=0,
    plot_args=None,
    loc="right",
    ax=None,
    show=True,
):
    # here numpy can be false and I get the dim with (pyvips.image.get("width"), pyvips.image.get("height"))
    # should also allow pyvips images as slide
    if not plot_args:
        plot_args = {
            "color": None,
            "size": (10, 10),
            "title": "",
            "alpha": 0.6,
            "cmap": plt.cm.plasma,
            "vmin": None,
            "vmax": None,
        }
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=plot_args["size"], layout="constrained")
    ax.set_title(plot_args["title"], size=20)
    ax.axis("off")

    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if res_to_view > slide.level_count:
        print(
            f"downsampling level={res_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        res_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]
    wsi = slide.read_region((0, 0), res_to_view, slide.level_dimensions[res_to_view])
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]
    dim_at_res_to_view = slide.level_dimensions[res_to_view]
    down_at_res_to_view = slide.level_downsamples[res_to_view]
    ax.imshow(wsi, aspect="equal")

    list_pos = pd.Series(tile_feat.index).to_list()
    if plot_args["vmin"] is None:
        plot_args["vmin"] = tile_feat[name].min()
    if plot_args["vmax"] is None:
        plot_args["vmax"] = tile_feat[name].max()

    norm = plt.Normalize(plot_args["vmin"], plot_args["vmax"])
    sm = plt.cm.ScalarMappable(cmap=plot_args["cmap"], norm=norm)
    sm.set_array([])

    for idx in list_pos:
        x, y = get_x_y_to(
            (tile_feat["x"][idx], tile_feat["y"][idx]),
            dim_at_analyse_level,
            dim_at_res_to_view,
            integer=True,
        )
        width, height = get_size_to(
            (tile_feat["w"][idx], tile_feat["h"][idx]),
            down_at_analyse_level,
            down_at_res_to_view,
            integer=True,
        )
        feat = tile_feat[name][idx]
        plot_seed = (x, y)
        color = plot_args["cmap"](norm(feat))
        patch = patches.Rectangle(
            plot_seed,
            width,
            height,
            fill=True,
            facecolor=color,
            edgecolor=plot_args["color"],
            alpha=plot_args["alpha"],
        )
        ax.add_patch(patch)

    ax.set_title(plot_args["title"], size=20)
    ax.axis("off")
    if loc:
        plt.colorbar(sm, ax=ax, label=name, location=loc)
    if show:
        plt.show()


def visualise_cut(
    slide,
    xywh,
    analyse_level=0,
    res_to_view=0,
    plot_args={"color": "red", "size": (10, 10), "title": ""},
    ax=None,
    show=True,
):
    """
    Plots the patches you are going to extract from the slide. So that they
    appear as red boxes on the lower resolution of the slide.
    Args:
        slide : str or openslide object.
        list_pos : list of parameters to extract tiles from slide.
        res_to_view : integer (default: None) resolution at which to
                      view the patch extraction.
        plot_args : dictionnary for any plotting argument.
    """
    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if res_to_view > slide.level_count:
        print(
            f"downsampling level={res_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        res_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]
    wsi = slide.read_region((0, 0), res_to_view, slide.level_dimensions[res_to_view])
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]
    dim_at_res_to_view = slide.level_dimensions[res_to_view]
    down_at_res_to_view = slide.level_downsamples[res_to_view]
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=plot_args["size"], layout="constrained")
    ax.imshow(wsi, aspect="equal")
    for para in xywh:
        x, y, w, h = para
        x, y = get_x_y_to(
            (x, y), dim_at_analyse_level, dim_at_res_to_view, integer=True
        )
        w, h = get_size_to(
            (w, h), down_at_analyse_level, down_at_res_to_view, integer=True
        )
        plot_seed = (x, y)
        patch = patches.Rectangle(
            plot_seed, w, h, fill=False, edgecolor=plot_args["color"]
        )
        ax.add_patch(patch)
    ax.set_title(plot_args["title"], size=20)
    ax.axis("off")
    if show:
        plt.show()
