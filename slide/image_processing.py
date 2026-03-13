# General libraries
from functools import partial
import numpy as np
import warnings
import openslide

# Image processing modules
from skimage.color import rgb2gray
from skimage.feature._hog import _hog_channel_gradient, hog
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, square

# Project modules
from slide.reader import OpenWSI
from bioformat.reader import OpenOME
from slide.utils import get_x_y_to

# Defining valid statistic function
valid_stat_func = ["mean", "quantile", "min", "max", "channel_std"]
valid_agg_func = ["sum", "mean", "max", "min", "quantile"]


def clear_border(
    img: np.ndarray, margin: tuple[int, int] | int, border_value: int = 1
) -> np.ndarray:
    """
    Clear the border of an image defined, whose width is defined by the margin parameter.
    It sets the values within the border to `border_value`.

    Args:
        img:            np.ndarray, image on which to clear the border.
        margin:         tuple of int or int, margin defining the width of the border on each dimension.
        border_value:   int, value to set the border to, default=0.

    Returns:
        clean_img:      np.ndarray, image on which the border was cleared.

    """

    # Parsing margin argument
    if isinstance(margin, int):
        margin = (margin, margin)

    # Retrieving the shapes
    r, c = img.shape
    mr, mc = margin

    # Clearing the border
    clean_img = img.copy()
    clean_img[:mr, :] = border_value
    clean_img[r - mr :, :] = border_value
    clean_img[:, :mc] = border_value
    clean_img[:, c - mc :] = border_value

    # Returning the cleaned image
    return clean_img


def check_borders_correct(
    img: np.ndarray, point: tuple[int, int] | int
) -> tuple[int, int]:
    """
    Check if a point's coordinates are within the image.
    Args:
        img:                np.ndarray, image on which we want to check if the point's coordinates are within the image.
        point:              tuple of int, point's coordinates to check

    Returns:
        updated_point:      tuple of int, new point's coordinates after checking and replacing its coordinates at the
                            image border if necessary.

    """

    # Retrieving image shape
    shape = img.shape

    # Check if point is inside img otherwise relocate within image
    if point[0] < 0 or point[1] < 0 or point[0] > shape[0] or point[1] > shape[1]:
        x, y = point
        x = max(0, x)
        y = max(0, y)
        x = min(shape[0], x)
        y = min(shape[1], y)
        warnings.warn("Invalid point: {}, corrected to {}".format(point, (x, y)))
        point = (x, y)

    # Returning point
    return point


def slice_img(
    img: np.ndarray,
    point_0: tuple[int, int] | list[int],
    point_1: tuple[int, int] | list[int] = None,
) -> np.ndarray:
    """
    Slice the image between point_0 and point_1, and return the corresponding image extracted.
    If point_1 is None, then it returns the image at point_0.

    Args:
         img:           np.ndarray, image to slice.
         point_0:       tuple or list of int, first point's coordinate at which to slice.
         point_1:       tuple of or list of int, second point's coordinate at which to slice.

    Returns:
        sliced_img:     np.ndarray, image extracted between point_0 and point_1 or at point_0.
    """

    # Check first point's coordinates
    x_0, y_0 = check_borders_correct(img=img, point=point_0)

    # If no second point, evaluate the image at first image
    if point_1 is None:
        sliced_img = img[x_0, y_0]

    # Otherwise, extract the image slice between the two points
    else:

        # Check last point's coordinates
        x_1, y_1 = check_borders_correct(img=img, point=point_1)

        # Check if first point is before second point
        assert x_0 <= x_1, "Invalid x_axis slicing, point_0: {} and point_1: {}".format(
            point_0, point_1
        )
        assert y_0 <= y_1, "Invalid y_axis slicing, point_0: {} and point_1: {}".format(
            point_0, point_1
        )

        # Extracting image slice
        sliced_img = img[x_0:x_1, y_0:y_1]

    # Returning the sliced image
    return sliced_img


def mask_percentage(
    mask: np.ndarray,
    point: tuple[int, int] | list[int],
    radius: int | tuple[int] | np.ndarray,
    mask_tolerance: float = 0.5,
) -> bool:
    """
    Compute the percentage of overlap between the patch, defined by `point` as its center and `radius` as its width
    and height, and the mask. If the percentage is higher than `mask_tolerance`, return True.

    Args:
        mask:           np.ndarray, mask on which to compute the overlap with the patch.
        point:          tuple or list of int, center point's coordinates of the patch on which to compute the overlap
                        with the mask.
        radius:         int, radius of the patch on which to compute the overlap with the mask, in pixels.
        mask_tolerance: float, percentage threshold of overlap between the mask and the patch center, default=0.5.

    Returns:
        is_accepted:    bool, whether the patch is accepted or not.

    """
    # Extracting mask value at patch location
    sub_mask = slice_img(
        img=mask,
        point_0=(np.array(point) - np.array(radius)).tolist(),
        point_1=(np.array(point) + np.array(radius) + 1).tolist(),
    )

    # Compute percentage of overlapping in the patch
    score = sub_mask.sum() / (sub_mask.shape[0] * sub_mask.shape[1])

    # Returning if the patch is valid or not
    return score > mask_tolerance


# contrast enhancer see tiatoolbox
def compute_luminosity_mask(
    slide: OpenWSI | OpenOME | openslide.OpenSlide,
    mask_level: int,
    margin: int | tuple[int, int] = 0,
    intensity_thresh: tuple[float, float] = (0.1, 0.8),
    neighborhood_size: int = 2,
) -> np.ndarray:
    """Get tissue mask based on the luminosity of the input image at the level `mask_level`, and removing the slide's border defined by `margin`.

    Args:
        slide:                  openslide.OpenSlide like, slide to process.
        mask_level:             int, level of resolution of the mask.
        margin:                 int or tuple of int, margin defining the border to remove, default=0.
        intensity_thresh:       float, Luminosity threshold for filtering the pixels in tissue area (high value are glass).
        neighborhood_size:      int, size of the neighborhood for computing the opening and closing, default=2.

    Returns:
        mask:  np.ndarray Binary tissue mask.

    """
    # Extracting region from WSI slide at right level
    img = slide.read_region(
        location=(0, 0), level=mask_level, size=slide.level_dimensions[mask_level]
    )

    # Convert to numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)[:, :, :3]

    # Parsing to grayscale and clearing border
    img_gray = rgb2gray(img)
    img_gray = clear_border(img_gray, margin=margin)

    mask = opening(
        closing(
            np.logical_and(
                img_gray < max(intensity_thresh), img_gray > min(intensity_thresh)
            ),
            footprint=square(neighborhood_size),
        ),
        footprint=square(neighborhood_size),
    )

    # check it's not empty
    if mask.sum() == 0:
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    return mask


def compute_otsu_mask(
    slide: OpenWSI | OpenOME | openslide.OpenSlide,
    mask_level: int,
    margin: int | tuple[int, int] = 0,
    intensity_thresh: tuple[float, float] = (0.1, 0.98),
    neighborhood_size: int = 2,
) -> np.ndarray:
    """
    Create a foreground mask for the slide at the level `mask_level`, and removing the slide's border defined by `margin`.

    Args:
        slide:                  OpenWSI or openslide.OpenSlide, slide to process.
        mask_level:             int, level of resolution of the mask.
        margin:                 int or tuple of int, margin defining the border to remove, default=0.
        intensity_thresh:       tuple of float, intensity threshold for filtering the pixels before computing Otsu,
                                thresholding, default=(0.1, 0.98).
        neighborhood_size:      int, size of the neighborhood for computing the opening and closing, default=2.

    Returns:
        mask:  np.ndarray Binary tissue mask.
    """
    # Extracting region from WSI slide at right level
    img = slide.read_region(
        location=(0, 0), level=mask_level, size=slide.level_dimensions[mask_level]
    )

    # Convert to numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)[:, :, :3]

    # Parsing to grayscale and clearing border
    img_gray = rgb2gray(img)
    img_gray = clear_border(img_gray, margin=margin)
    img_gray = img_gray.flatten()

    # Computing Otsu threshold on filtered intensity
    pixels_int = img_gray[
        np.logical_and(
            img_gray > min(intensity_thresh), img_gray < max(intensity_thresh)
        )
    ]
    t = threshold_otsu(pixels_int)

    # Create a mask for filtering the image based on Otsu threshold with opening (remove bright spots and dark cracks)
    # And closing (remove dark spots and bright cracks) post-processing
    mask = opening(
        closing(
            np.logical_and(img_gray < t, img_gray > min(intensity_thresh)).reshape(
                img.shape[:-1]
            ),
            footprint=square(neighborhood_size),
        ),
        footprint=square(neighborhood_size),
    )

    # check it's not empty
    if mask.sum() == 0:
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    # Returning the mask
    return mask


def get_patch_mask(
    coords: np.ndarray, mask_dim: tuple[int, int], slide_dim: tuple[int, int]
) -> np.ndarray:
    """
    Get a mask for the patches, where the values correspond to the patches' index or -1 if it is the background.
    The patches are placed on the mask matching their coordinates on the slide.

    Args:
        coords:         np.ndarray, array of size (n_tiles, 5) containing the coordinates (x, y, size_x, size_y) of the
                        patches.
        mask_dim:       tuple of int, dimension of the mask.
        slide_dim:      tuple of int, dimension of the slide.

    Returns:
        patch_mask:     np.ndarray, array of size mask_dim containing the mask of the patches.

    """

    # Retrieving downsample factor
    downsample_factor = (slide_dim[0] / mask_dim[0], slide_dim[1] / mask_dim[1])

    # Initialization of the mask for the patches, -1 corresponds to background
    patch_mask = -np.ones(mask_dim)

    # For each patch, put its index at its corresponding position in the mask
    for i, (x, y, size_x, size_y) in enumerate(coords):
        mask_patch_size = (
            max(1, int(size_x / downsample_factor[0])),
            max(1, int(size_y / downsample_factor[1])),
        )
        x, y = get_x_y_to(
            point=(x, y), dim_from=slide_dim, dim_to=mask_dim, integer=True
        )
        patch_mask[x : (x + mask_patch_size[0]), y : (y + mask_patch_size[1])] = i

    # Returning mask for the patches
    return patch_mask

# TODO: Make faster filtering function 

def compute_channel_gradient(image: np.ndarray, channel_axis: int = None):
    """
    Compute the gradient of an image using each channel, and obtaining the overall gradient by taking the maximum over
    the channels' gradients.

    Args:
        image:              np.ndarray, image to compute gradient over.
        channel_axis:       int, channel axis to compute gradient over.

    Returns:
        g_row:              np.ndarray, the gradient of an image over the rows.
        g_col:              np.ndarray, the gradient of an image over the columns.

    """

    # Parsing image as type float
    image = image.astype("float", copy=False)

    # Retrieving multichannel args
    multichannel = False
    if channel_axis is not None:
        multichannel = True

    # Computing gradient per channel
    if multichannel:

        # Initialization of arrays for image gradients
        g_row_by_ch = np.empty_like(image, dtype=image.dtype)
        g_col_by_ch = np.empty_like(image, dtype=image.dtype)
        g_magn = np.empty_like(image, dtype=image.dtype)

        # Compute the gradients for each channel and the global gradient's magnitude
        for idx_ch in range(image.shape[channel_axis]):
            (
                g_row_by_ch[:, :, idx_ch],
                g_col_by_ch[:, :, idx_ch],
            ) = _hog_channel_gradient(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(
                g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch]
            )

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = g_magn.argmax(axis=channel_axis)
        rr, cc = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            indexing="ij",
            sparse=True,
        )
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]

    # Otherwise, compute overall gradient
    else:
        g_row, g_col = _hog_channel_gradient(image)

    # Returning the gradients
    return g_row, g_col


def compute_hog(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (1, 1),
    use_grayscale: bool = False,
    hog_image: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Compute the HOG descriptors and image for a given image.

    Args:
        image:                  np.ndarray, image to analyze.
        orientations:           int, number of orientation bins, default=9.
        pixels_per_cell:        tuple of int, number of pixels per cell, default=(16, 16).
        cells_per_block:        tuple of int, number of cells per block, default=(1, 1).
        use_grayscale:          bool, whether to use the grayscale image or the RGB image, default=False.
        hog_image:              bool, whether to compute the HOG image along the HOG features or not, default=False.

    Returns:
        feats:                  np.ndarray, HOG descriptors.
        hog_image:              np.ndarray, HOG image.

    """

    # Convert image to grayscale and define channel axis
    if use_grayscale:
        image = rgb2gray(image)
        channel_axis = None
    else:
        channel_axis = -1

    # Parsing visualize default arg
    visualize = False
    if hog_image:
        visualize = True

    # Compute HOG descriptors
    res = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        channel_axis=channel_axis,
    )

    # Returning the HOG features and image
    return res


def compute_channel_std(
    img_list: list[np.ndarray] | np.ndarray, aggregate: str = "mean", q: float = 0.5
):
    """
    Compute the standard deviation between the channel for each pixel in the image and for each image in the list.
    Aggregate the results with the `aggregate` argument.

    Args:
        img_list:       list of np.ndarray or np.ndarray, list containing the images or array containing the flattened
                        images (n_pixels, 3) in its first dimension.
        aggregate:      str, aggregate function, default='mean'.
        q:              float, quantile to compute on the standard deviation for aggregation if aggregate='quantile'.

    Returns:
        stat_values:    np.ndarray, array of size (n_images, 1) containing the images aggregated statistics.

    """
    # Retrieving the aggregate function
    assert hasattr(np, aggregate), (
        "Aggregate function {} not implemented.".format(aggregate)
        + f"Please choose a valid argument between: {', '.join(valid_agg_func)}."
    )
    agg_func = getattr(np, aggregate)
    if aggregate == "quantile":
        agg_func = partial(agg_func, q=q)

    # Computing channel std
    channel_std = np.array(img_list).std(axis=-1)

    # Aggregated the results across images in the list
    if np.isnan(channel_std).any():
        stat_values = np.apply_along_axis(
            arr=channel_std, func1d=lambda x: agg_func(x[~np.isnan(x)]), axis=1
        )
    else:
        stat_values = agg_func(
            channel_std[~np.isnan(channel_std)].reshape(len(img_list), -1), axis=-1
        )

    # Returning the results
    return stat_values.reshape(len(img_list), -1)


def hog_selection(
    img_list: list[np.ndarray],
    hog_thresh: float = 0.5,
    hog_std_thresh: float = None,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (1, 1),
    use_grayscale: bool = False,
):
    """
    Compute HOG descriptors and filter the images contained in `img_list` based on the mean and std HOG values:
    mean hog >= hog thresh & std hog >= hog std thresh.

    Args:
        img_list:               list of arrays, list containing the images to filter based on their HOG values.
        hog_thresh:             float, threshold to overpass for the HOG descriptors' mean value, default=0.5.
        hog_std_thresh:         float, threshold to overpass for the HOG descriptors' std value, default=None.
        orientations:           int, number of orientation bins to use to compute the HOG descriptors, default=9.
        pixels_per_cell:        tuple of ints, number of pixels per cell, default=(16,16).
        cells_per_block:        tuple of ints, number of cells per block, default=(1,1).
        use_grayscale:          bool, whether to use the grayscale images over the RGB image, default=False.

    Returns:
        idx_selected:           np.ndarray, array of size (n_img_selected) containing the indices of the selected images.

    """

    # Initialization of a list to store hog values
    hog_list = []

    # Computing HOG descriptors
    for i in range(len(img_list)):
        _, hog_img = compute_hog(
            image=img_list[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            use_grayscale=use_grayscale,
            hog_image=True,
        )
        hog_list.append(hog_img)

    # Filtering the patches based on hog mean values
    hog_filter = (
        np.array(hog_list).reshape(len(hog_list), -1).mean(axis=-1) >= hog_thresh
    )

    # Filtering the patches based on hog std values as well
    if hog_std_thresh is not None:
        hog_filter &= (
            np.array(hog_list).reshape(len(hog_list), -1).std(axis=-1) >= hog_std_thresh
        )

    # Returning the indices of the selected patch
    return np.argwhere(hog_filter).squeeze()


def patch_mean_pooling(
    img: np.ndarray, patch_size: int | tuple[int, int]
) -> np.ndarray:
    """
    Average pooling over patches of size `patch_size` in the given image.

    Args:
        img:            np.ndarray, array of size (n_row, n_col, n_channels) containing the image to average over.
        patch_size:     int or tuple of ints, size of the patches to average over.

    Returns:
        mean_img:       np.ndarray, array of size (n_patch_row, n_patch_col, n_channels) containing the image averaged
                        over the patches.

    """

    # Parsing patch size
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    # Retrieving number of patches per row and columns
    n_patch_row, n_patch_col = (
        img.shape[0] // patch_size[0],
        img.shape[1] // patch_size[1],
    )

    # Initialization of the local average image
    mean_img = np.zeros((n_patch_row, n_patch_col, img.shape[-1]))

    # For each patch, compute the average of the pixels
    for i in range(n_patch_row):
        for j in range(n_patch_col):
            patch = img[
                i * patch_size[0] : min((i + 1) * patch_size[0], img.shape[0]),
                j * patch_size[1] : min((j + 1) * patch_size[1], img.shape[1]),
            ].reshape(-1, img.shape[-1])
            if sum(~np.isnan(patch).any(axis=-1)) > 0:
                mean_img[i, j, :] = patch[~np.isnan(patch).any(axis=-1), :].mean(axis=0)
            else:
                mean_img[i, j, :] = np.nan

    # Returning the averaged image per patch
    return mean_img


def color_filter_selection(
    img_list: list[np.ndarray] | np.ndarray,
    color_thresh: tuple[int, int, int] | int,
    stat_fct: str = "quantile",
    q: float = 0.05,
    remove_background: bool = True,
    background_thresh: tuple[int, int, int] | int = 245,
    local_average: bool = True,
    filter_size: tuple[int, int] | int = 16,
    filter_all: bool = False,
    background_average: bool = True,
    aggregate: str = "mean",
) -> list[int]:
    """
    Filter the images contained in `img_list` based on their mean value if q is None or the `q`-th quantile value
    otherwise. This statistic is computed for each channel (R,G,B) and compared to the color threshold value.

    Args:
        img_list:           list of np.ndarray, list containing the images to filter based on their color threshold.
        color_thresh:       tuple of int or int, color threshold for each channel (R,G,B).
        stat_fct:           str, statistic to compute and compare its value to the threshold, default='quantile'.
        q:                  float, quantile to compute for comparing to the threshold, default=0.1.
        remove_background:  bool, whether to remove the background before computing the color statistic threshold.
        background_thresh:  tuple of int or int, color threshold for detecting background for each channel (R,G,B),
                            default=245.
        local_average:      bool, whether to compute the local average before computing the statistic value, default=False.
        filter_size:        tuple of int or int, size of the patches to perform local averaging before computing the
                            statistic, default=None, meaning no local averaging is performed beforehand.
        filter_all:         bool, whether the criterion on the color threshold has to be met for all channel or only one,
                            default=False.
        background_average: bool, whether to compute the background on the patche average before or not, default=True.
        aggregate:          str, aggregate function for the channel standard deviation aggregation, default='mean'.

    Returns:
        idx_selected:       list of int, list containing the indices of the selected images.

    """

    # Parsing color threshold argument
    if isinstance(color_thresh, int):
        color_thresh = (color_thresh, color_thresh, color_thresh)
    if remove_background and isinstance(background_thresh, int):
        background_thresh = (background_thresh, background_thresh, background_thresh)

    # Checking valid value of color_thresh
    assert (
        color_thresh is not None
    ), f"No color threshold was given to filter the patches. \nPlease specify a `color_thresh` value."

    # Retrieving the statistic to compute
    assert stat_fct in valid_stat_func, (
        f"The statistic function given is not implemented. "
        + f"\nPlease specify a valid `stat_fct` value among: {', '.join(valid_stat_func)}."
    )

    # Retrieving the statistic function
    if stat_fct == "quantile":
        stat_fn = partial(np.quantile, q=q, axis=-2)
    elif stat_fct == "channel_std":
        stat_fn = partial(compute_channel_std, aggregate=aggregate)
    else:
        stat_fn = partial(getattr(np, stat_fct), axis=-2)

    # Removing background on image before computing the local average
    if remove_background and not background_average:
        clean_imgs = []
        for img in img_list:
            clean_img = img.copy().astype(float)
            clean_img[(clean_img >= background_thresh).any(axis=-1), :] = None
            clean_imgs.append(clean_img)
        img_list = clean_imgs

    # Local averaging of the input
    if local_average:
        mean_imgs = []
        for img in img_list:
            mean_imgs.append(patch_mean_pooling(img=img, patch_size=filter_size))
        img_list = mean_imgs

    # Removing background and compute the statistic for each cleaned image due to different dimensions
    if remove_background and background_average:
        stat_values = []
        for img in img_list:
            clean_img = img[(img < background_thresh).all(axis=-1), :]
            stat_values.append(stat_fn(clean_img.reshape(1, -1, clean_img.shape[-1])))
        stat_values = np.array(stat_values).squeeze(-2)

    # Otherwise, compute the statistic on all images in one shot since they have same dimensions
    else:
        stat_values = stat_fn(
            np.array(img_list).reshape(len(img_list), -1, img_list[0].shape[-1])
        )

    # Filter for the images based on the channels' minimum color threshold
    img_filter = stat_values >= np.array(color_thresh)
    if filter_all:
        img_filter = img_filter.all(axis=-1)
    else:
        img_filter = img_filter.any(axis=-1)

    # Retrieving the indices selected
    idx_selected = np.argwhere(img_filter).squeeze(-1).tolist()

    # Returning selected indices
    return idx_selected


def filter_gray_img(
    img_list: list[np.ndarray] | np.ndarray,
    std_thresh: int,
    remove_background: bool = True,
    background_thresh: int | tuple[int, int, int] = 245,
    local_average: bool = True,
    filter_size: int | tuple[int, int] = 16,
    aggregate: str = "mean",
):
    """
    Filter images containing gray pixels on average (channel std < std_thresh) after having removed the background.

    Args:
        img_list:               list of np.ndarrays or np.ndarrays, list containing the images to be filtered.
        std_thresh:             int, standard deviation across channel threshold.
        remove_background:      bool, whether to remove the background before computing the channel std, default=True.
        background_thresh:      int or tuple of int, color threshold for each channel (R,G,B), default=245.
        local_average:          bool, whether to compute the local average before computing the channel std, default=True.
        filter_size:            int or tuple of int, size of the patches on which to perform local averaging before
                                computing the channel std, default=16.
        aggregate:              str, aggregate function for the channel standard deviation aggregation across patches if
                                local averaging, default='mean'.

    Returns:

    """

    # Color filter based on channel std
    idx_selected = color_filter_selection(
        img_list=img_list,
        color_thresh=std_thresh,
        stat_fct="channel_std",
        remove_background=remove_background,
        background_thresh=background_thresh,
        local_average=local_average,
        filter_size=filter_size,
        aggregate=aggregate,
    )

    # Returning selected indices
    return idx_selected
