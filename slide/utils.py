import os
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide
from skimage.color import rgb2gray
from skimage.morphology import square, closing, opening
from skimage.filters import threshold_otsu
import warnings
import itertools
import osfile.manager as fm
from bioformat.utils import check_is_ome
from bioformat.reader import OpenOME
from slide.reader import OpenWSI
import json
import h5py
import pandas as pd


OPENSLIDE_READABLE_FORMATS = [
    ".svs",
    ".tif",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".tiff",
    ".svslide",
    ".bif",
]


def openslide_metadata_to_xml(path):
    return "Not implemented yet"


def get_slide_name_extension(path):
    name, ext = os.path.splitext(os.path.basename(path))
    return name, ext


def check_to_use_openslide(path):
    _, ext = get_slide_name_extension(path)
    use_openslide = ext.lower() in OPENSLIDE_READABLE_FORMATS
    return use_openslide


def get_slide_reader(path):
    if check_is_ome(path):
        return OpenOME
    elif check_to_use_openslide(path):
        return OpenWSI
    else:
        raise TypeError("file with wrong extension")


# get metadata of slide (uses openslide to handle the image, must be compatible)
def get_openslide_pyramid_info(slide, verbose=False):
    # assert slide format
    if isinstance(slide, str):
        slide = OpenSlide(slide)
    slide_prop = dict(slide.properties)
    infos = {
        "level_count": slide.level_count,
        "size_0": (slide.level_dimensions[0][1], slide.level_dimensions[0][0]),
        "res_0": (
            float(slide_prop["openslide.mpp-y"]),
            float(slide_prop["openslide.mpp-x"]),
        ),  # YX = HW
        "objective": int(slide_prop["openslide.objective-power"]),
        "downsampling": [
            int(slide_prop[f"openslide.level[{l}].downsample"])
            for l in range(slide.level_count)
        ],
    }
    if verbose:
        print(infos)
    return infos


# make thumbnail for opensilde
def make_openslide_thumbnail(
    scr, dst=None, level=0, format="ndpi", grayscale=False, verbose=True
):
    files, cnt = fm.findFile(scr, format, fileExtensions=True)
    if verbose:
        print(f"{cnt} WSI.{format} were found in source dir: {scr}")
    # Make dst folders if not aready exist
    if not dst:
        dst = os.path.join(scr, f"visu_slide")
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    # Iterate over files found in scr
    for file in files:
        filename, _ = os.path.splitext(file)
        tag = os.path.basename(filename)
        # Slide view
        # Get WSI (use function get_slice, get image)
        wsi_np = get_slide_whole(file, level=level)
        if grayscale:
            wsi_np = (rgb2gray(wsi_np).astype("float32") - 1) * -1
        wsi_size = wsi_np.shape
        wsi_aspect = wsi_size[1] / wsi_size[0]
        fig, ax = plt.subplots(
            1, 1, figsize=(12, 12 / wsi_aspect), layout="constrained"
        )
        ax.set_title(f"{tag} at level={level}")
        ax.imshow(wsi_np)
        ax.axis("off")
        fig.savefig(os.path.join(dst, f"{tag}_slide_level_{level}.png"))
        plt.close()
        if verbose:
            print(f"slide thumbnail saved in {dst}")
        # Auto Mask view
        # get binary mask for tissue segementation (use function make_auto_mask)
        mask = make_auto_mask(file, mask_level=level)
        fig, ax = plt.subplots(
            1, 1, figsize=(12, 12 / wsi_aspect), layout="constrained"
        )
        ax.set_title(f"{tag} tissue mask at level={level}")
        ax.imshow(mask)
        ax.axis("off")
        fig.savefig(os.path.join(dst, f"{tag}_mask_level_{level}.png"))
        plt.close()
        if verbose:
            print(f"mask thumbnail saved in {dst}")


def get_slide_whole(slide, level=None, numpy=True):
    """
    Return whole image at a certain level.
    Args:
        slide : String or openslide object from which we extract.
        level : Integer, by default None. If None the value is set to
                the maximum level minus one of the slide. Level at which
                we extract.
        numpy : Boolean, by default True, wether or not to convert the output to numpy array instead
                of PIL image.
    Returns:
        A numpy array or PIL image corresponding the whole slide at a given
        level.
    """
    if isinstance(slide, str):
        slide = OpenSlide(slide)
    if level is None:
        level = slide.level_count - 1
    elif level > slide.level_count - 1:
        print(" level ask is too low... It was setted accordingly")
        level = slide.level_count - 1
    sample = slide.read_region((0, 0), level, slide.level_dimensions[level])
    if numpy:
        sample = np.array(sample)[:, :, 0:3]
    return sample


def get_slide_image(slide, para, numpy=True):
    """
    Returns cropped image given a set of parameters.
    You can feed a string or an openslide image.
    Args:
        slide : String or openslide object from which we extract.
        para : List of 5 integers corresponding to: [x, y, size_x_level, size_y_level, level]
        numpy : Boolean, by default True, wether or not to convert the output to numpy array instead
                of PIL image.
    Returns:
        A tile (or crop) from slide corresponding to para. It can be a numpy array
        or a PIL image.

    """
    if isinstance(para, dict):
        slide = OpenSlide(slide) if isinstance(slide, str) else slide
        slide = slide.read_region(
            (para["x"], para["y"]), para["level"], (para["xsize"], para["ysize"])
        )
        if numpy:
            slide = np.array(slide)[:, :, 0:3]
    else:
        if len(para) != 5:
            raise NameError("Not enough parameters...")
        slide = OpenSlide(slide) if isinstance(slide, str) else slide
        slide = slide.read_region((para[0], para[1]), para[4], (para[2], para[3]))
        if numpy:
            slide = np.array(slide)[:, :, 0:3]
    return slide


# rename get_new_size
def get_size_to(size, downsample_from, downsample_to, integer=True):
    """
    Returns:
        A tuple, or tuple like object of size 2 with integers corresponding
        to the new size at level level_to. Or size_to.
    """
    size_x, size_y = size
    scal = float(downsample_from / downsample_to)
    if integer:
        func_round = round
    else:
        func_round = lambda x: x
    size_x_new = func_round(float(size_x) * scal)
    size_y_new = func_round(float(size_y) * scal)
    size_to = size_x_new, size_y_new
    return size_to


# rename get_xy_to
def get_x_y_to(point, dim_from, dim_to, integer=True):
    """
    Returns:
        A tuple corresponding to the converted coordinates, point_l.
    """
    x_0, y_0 = point
    size_x_l = float(dim_to[0])
    size_y_l = float(dim_to[1])
    size_x_0 = float(dim_from[0])
    size_y_0 = float(dim_from[1])

    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0
    if integer:
        point_l = (round(x_l), round(y_l))
    else:
        point_l = (x_l, y_l)
    return point_l


def make_auto_mask(slide, mask_level, margin=(0, 0)):
    img = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
    if not isinstance(img, np.ndarray):
        img = np.array(img)[:, :, :3]
    img_gray = rgb2gray(img)
    img_gray = clear_border(img_gray, margin=margin)
    size = img_gray.shape
    img_gray = img_gray.flatten()
    pixels_int = img_gray[np.logical_and(img_gray > 0.1, img_gray < 0.98)]
    t = threshold_otsu(pixels_int)
    mask = opening(
        closing(
            np.logical_and(img_gray < t, img_gray > 0.1).reshape(size),
            footprint=square(2),
        ),
        footprint=square(2),
    )
    return mask


def clear_border(mask, margin):
    r, c = mask.shape
    mr, mc = margin
    mask[:mr, :] = 0
    mask[r - mr :, :] = 0
    mask[:, :mc] = 0
    mask[:, c - mc :] = 0
    return mask


def grid_blob(point_start, point_end, space):
    """
    Returns:
        List of coordinates of grid.
    """
    size_x, size_y = space
    list_x = range(point_start[0], point_end[0], size_x)
    list_y = range(point_start[1], point_end[1], size_y)
    return list(itertools.product(list_x, list_y))


def check_borders_correct(array_np, point):
    shape = array_np.shape
    if point[0] < 0 or point[1] < 0 or point[0] > shape[0] or point[1] > shape[1]:
        x, y = point
        x = max(0, x)
        y = max(0, y)
        x = min(shape[0], x)
        y = min(shape[1], y)
        warnings.warn("Invalid point: {}, corrected to {}".format(point, (x, y)))
        point = (x, y)
    return point


def pj_slice(array_np, point_0, point_1=None):
    """
    Returns:
        If point_1 is None, returns array_np evaluated in point_0,
        else returns a slice of array_np between point_0 and point_1.
    """
    x_0, y_0 = check_borders_correct(array_np, point_0)
    if point_1 is None:
        result = array_np[x_0, y_0]
    else:
        x_1, y_1 = check_borders_correct(array_np, point_1)
        if x_0 > x_1:
            warnings.warn(
                "Invalid x_axis slicing, \
                point_0: {} and point_1: {}".format(
                    point_0, point_1
                )
            )
        if y_0 > y_1:
            warnings.warn(
                "Invalid y_axis slicing, \
                point_0: {} and point_1: {}".format(
                    point_0, point_1
                )
            )
        result = array_np[x_0:x_1, y_0:y_1]
    return result


def mask_percentage(mask, point, radius, mask_tolerance=0.5):
    """
    Returns:
        A boolean. If True, keep, else discard
    """
    sub_mask = pj_slice(mask, point - radius, point + radius + 1)
    score = sub_mask.sum() / (sub_mask.shape[0] * sub_mask.shape[1])
    accepted = score > mask_tolerance
    return accepted


def get_bag_of_tiles(slide, xywh, res_to_view=0):
    bag = []
    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    for pos in xywh:
        x, y, w, h = pos
        tile = slide.read_region(location=(x, y), level=res_to_view, size=(w, h))
        if not isinstance(tile, np.ndarray):
            tile = np.array(tile[:, :, :3])
        bag.append(tile)

    if len(bag) == 1:
        return bag[0]
    else:
        return bag


def read_h5_coords(coords_path):
    with h5py.File(coords_path, "r") as f:
        attrs = dict(f["coords"].attrs)
        coords = f["coords"][:]
    return attrs, coords

def read_h5_features(embs_path):
    with h5py.File(embs_path, "r") as f:
        attrs = dict(f["features"].attrs)
        feats = f["features"][:]
    return attrs, feats


def read_csv_coords(coords_path):
    coords = pd.read_csv(coords_path)
    return coords.to_numpy()


def read_array_coords(coords_path):
    return 0


def save_csv():
    return "Not implemented yet"


def save_np():
    return "Not implemented yet"


def save_pickel():
    return "Not implemented yet"


def print_attrs(name, obj):
    print(f"Object: {name}")
    for key, value in obj.attrs.items():
        print(f"    Attribute - {key}: {value}")


def print_dict(dict, name=None):
    if name:
        print(f"In {name}: ")
    else:
        print(f"In Dictionary: ")
    for key, value in dict.items():
        print(f"    {key}: {value}")


# taken from https://github.com/mahmoodlab/TRIDENT
def save_h5(save_path, assets, attributes=None, mode="w"):
    """
    The `save_h5` function saves a dictionary of assets to an HDF5 file. This is commonly used to store
    large datasets or hierarchical data structures in a compact and organized format.

    Parameters:
    -----------
    save_path : str
        The path where the HDF5 file will be saved.
    assets : dict
        A dictionary containing the data to save. Keys represent dataset names, and values are NumPy arrays.
    attributes : dict, optional
        A dictionary mapping dataset names to additional metadata (attributes) to save alongside the data. Defaults to None.
    mode : str, optional
        The file mode for opening the HDF5 file. Options include 'w' (write) and 'a' (append). Defaults to 'w'.

    Returns:
    --------
    None
        The function writes data and attributes to the specified HDF5 file.

    Example:
    --------
    >>> assets = {'data': np.array([1, 2, 3]), 'labels': np.array([0, 1, 1])}
    >>> attributes = {'data': {'description': 'Numerical data'}}
    >>> save_h5("output.h5", assets, attributes)
    >>> # Saves datasets and attributes to "output.h5".
    """

    with h5py.File(save_path, mode) as file:
        for key, val in assets.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunk_shape,
                    dtype=data_type,
                )
                dset[:] = val
                if attributes is not None:
                    if key in attributes.keys():
                        for attr_key, attr_val in attributes[key].items():
                            try:
                                # Serialize if the attribute value is a dictionary
                                if isinstance(attr_val, dict):
                                    attr_val = json.dumps(attr_val)
                                # Serialize Nones
                                elif attr_val is None:
                                    attr_val = "None"
                                dset.attrs[attr_key] = attr_val
                            except:
                                raise Exception(
                                    f"WARNING: Could not save attribute {attr_key} with value {attr_val} for asset {key}"
                                )

            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val

