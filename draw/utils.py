# General libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def safe_cast(val: Any, to_type: callable, default: Any = None) -> Any:
    """
    Safe casting value `val` using `to_type` function and if encounters invalid value return `default`.
    Particularly useful when casting float to int when there is NaN values.

    Args:
        val:            Any, value to cast to a different type.
        to_type:        Callable, function used for casting the value.
        default:        Any, default value to return if casting fails, default=None.

    Returns:
        parsed_val:     Any, the value broadcasted to the desired type.

    """
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def get_idx_fig(flat_idx: int, n_cols: int, n_rows: int) -> int | tuple[int, int]:
    """
    Get figure's 1D or 2D index from the flattened index.

    Args:
        flat_idx:       int, index in the figure's flattened grid.
        n_cols:         int, figure's grid number of columns.
        n_rows:         int, figure's grid number of rows.

    Returns:
        i_fig:          int or tuple of int, figure's corresponding 1D or 2D index.

    """

    # Retrieving axis specifications
    if n_rows == 1:
        i_fig = flat_idx % n_cols
    elif n_cols == 1:
        i_fig = flat_idx % n_rows
    else:
        i_fig = (flat_idx // n_cols, flat_idx % n_cols)

    # Returning indices of figures
    return i_fig


def get_ax_fig(ax: np.ndarray | plt.Axes, flat_idx: int) -> int | tuple[int, int]:
    """
    Get figure's corresponding axes at index `flat_idx` if the array was flattened.

    Args:
        ax:             array of plt.Axes or plt.Axes, array containing the figure's axes or directly the figure's axes.
        flat_idx:       int, index in the figure's flattened grid.

    Returns:
        ax_:            int or tuple of int, figure's corresponding axes at the specified index if the array was flattened.

    """

    # Returning corresponding axis
    if isinstance(ax, plt.Axes):
        return ax
    else:
        i_fig = get_idx_fig(flat_idx, n_cols=ax.shape[1], n_rows=ax.shape[0])
        return ax[i_fig]


def clear_unused_axes(ax: np.ndarray, start_idx: int, last_idx: int = None):
    """
    Clear unused axes from index `start_idx` to `last_idx` in array of plt.Axes stored in `ax`.

    Args:
        ax:             np.ndarray of plt.Axes, array containing the figure's axes.
        start_idx:      int, flat index from which to start clearing the subfigures' axes.
        last_idx:       int, flat index at which to stop clearing the subfigures' axes, default=None, meaning it will
                        stop at the last subfigure.

    """

    # Parsing last index arg
    if last_idx is None:
        last_idx = ax.shape[0] * ax.shape[1] - 1

    # Removing axis of unused figures
    for j in range(start_idx, last_idx + 1):

        # Retrieving corresponding subfigure and remove its axes
        ax_ = get_ax_fig(ax=ax, flat_idx=j)
        ax_.axis("off")

# make text_box_height match the number of line of text
def add_annotation_box(
    img: np.ndarray,
    text: list[str],
    title: str = "",
    text_y_spacing: int = 25,
    text_box_height: int = 150,
    text_box_width: int = 350,
    text_box_color: tuple[int, int, int] | str = None,
    alpha=0.25,
    title_fontscale: float = 0.75,
    thickness: int = 1,
    text_fontscale: float = 0.5,
    color: str | tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Add an annotation box to the image filled with the text contained in the list `text`, each element being a line in
    the annotation box.

    Args:
        img:                np.ndarray, image to which to add the annotation box.
        text:               list of str, annotations to add to the image.
        title:              str, title of the annotation box, default=''.
        text_y_spacing:     int, pixel spacing between the lines in the annotation box, default=25.
        text_box_height:    int, height of the annotation box, default=150.
        text_box_width:     int, width of the annotation box, default=350.
        text_box_color:     int or str, color of the annotation box, default=None, meaning it will be added without
                            background.
        alpha:              float, transparency of the annotation box, default=0.25.
        title_fontscale:    float, fontscale of the title in the annotation box, default=0.75.
        thickness:          int, thickness of the text in the annotation box, default=1.
        text_fontscale:     float, fontscale of the text in the annotation box, default=0.5.
        color:              tuple[int, int, int] or str, color of the text in the annotation box,
                            default=(255, 255, 255) corresponding to white writing.

    Returns:
        img:                np.ndarray, image to which the annotation box was added.

    """

    # Computing x offset
    text_x_offset = int(img.shape[0] * 0.03)

    # Adding background transparency
    if text_box_color is None:
        img[:text_box_height, :text_box_width] = (
            img[:text_box_height, :text_box_width] * alpha
        ).astype(np.uint8)
    else:
        # Create an overlay of the same size as the ROI, filled with the background color
        roi = img[:text_box_height, :text_box_width].copy()
        text_box = np.full_like(roi, text_box_color, dtype=np.uint8)
        overlay = cv2.addWeighted(text_box, 1 - alpha, roi, alpha, 0)
        img[:text_box_height, :text_box_width] = overlay

    # Adding title
    cv2.putText(
        img,
        text=title,
        org=(text_x_offset, text_y_spacing),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=title_fontscale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Add annotation
    for i, line in enumerate(text):
        cv2.putText(
            img,
            text=line,
            org=(text_x_offset, text_y_spacing * (i + 2)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=text_fontscale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # Returning the annotated image
    return img
