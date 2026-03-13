import os
import numpy as np
from openslide import OpenSlide
import itertools
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


def get_slide_name_extension(path) -> tuple[str, str]:
    """Get slide extension form path.

    Args:
        path (str):
            path to slide

    Returns:
        tuple:
            (name, ext): basename and extension of slide
    """
    name, ext = os.path.splitext(os.path.basename(path))
    return name, ext


def check_to_use_openslide(path):
    """Check if slide can be open with openslide.Openslide.

    Args:
        path (str):
            path to slide

    Returns:
        bool:
            is openslide.Openslide compatible
    """
    _, ext = get_slide_name_extension(path)
    use_openslide = ext.lower() in OPENSLIDE_READABLE_FORMATS
    return use_openslide


def get_slide_reader(path):
    """
    Get proper reader tool to open slide.

    Args:
        path (str):
            path to slide

    Returns:
        class:
            slide reader tool.
    """

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


def grid_blob(point_start, point_end, space):
    """
    Returns:
        List of coordinates of grid.
    """
    size_x, size_y = space
    list_x = range(point_start[0], point_end[0], size_x)
    list_y = range(point_start[1], point_end[1], size_y)
    return list(itertools.product(list_x, list_y))


def get_bag_of_tiles(slide, xywh, res_to_view=0):
    """
    Returns:
        List of tiles at xywh, extracted from slide.
    """
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


def read_h5_coords(coords_path, mode: str = "r") -> tuple[dict, np.ndarray]:
    """
    Get coordinates of the tiles from a HDF5 .h5 file.

    Args:
        coords_path:    str, path to the .h5 file containing the coordinates of the tile.

    Returns:
        attrs:          dict, dictionary containing the attributes of the coordinates of the tiles.
        coords:         np.ndarray, array containing the coordinates of the tiles.

    """
    with h5py.File(coords_path, mode) as f:
        attrs = dict(f["coords"].attrs)
        coords = f["coords"][:]
    return attrs, coords


def read_h5_features(embs_path: str, mode: str = "r") -> tuple[dict, np.ndarray]:
    """
    Get embeddings and attributes from a WSI embedding file.

    Args:
        embs_path:      str, path to the WSI embedding file.
        wsi_enc:        str, level of WSI encoding used, default='tile'.

    Returns:
        attrs:      dict, dictionary containing the corresponding WSI attributes.
        feats:      np.ndarray, size (n_tiles, dim_feats), array containing the WSI embeddings.

    """
    # Defining key for retrieving the embeddings in h5 file
    feat_key = "features"

    # Retrieving attributes and features
    with h5py.File(embs_path, mode) as f:
        attrs = dict(f[feat_key].attrs)
        feats = f[feat_key][:]

    # Returning attributes and features
    return attrs, feats


def read_csv_coords(coords_path: str) -> np.ndarray:
    """
    Get coordinates from a CSV file found at coords_path, including the .csv extension.

    Args:
        coords_path:    str, path to the CSV file, including the .csv extension.

    Returns:
        coords:         np.ndarray, array containing the coordinates of the tiles.

    """
    return pd.read_csv(coords_path).to_numpy()


def read_array_coords(path):
    return "Not implemented yet"


def save_csv(path):
    return "Not implemented yet"


def save_np(path):
    return "Not implemented yet"


def save_pickel(path):
    return "Not implemented yet"


def print_attrs(obj, name=None):
    """Print attributes from an object

    Returns:
        None
    """
    print(f"Object: {name}")
    for key, value in obj.attrs.items():
        print(f"    Attribute - {key}: {value}")


def print_dict(dict, name=None):
    """Print dictionary content

    Returns:
        None
    """
    if name:
        print(f"In {name}: ")
    else:
        print(f"In Dictionary: ")
    for key, value in dict.items():
        print(f"    {key}: {value}")


# taken from https://github.com/mahmoodlab/TRIDENT
def save_h5(save_path, assets, attributes=None, mode="a"):
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
                if data_type == object:  # when saving arrays of str for instance
                    data_type = h5py.string_dtype(encoding="utf-8")
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
                            except Exception as e:
                                print(e)
                                # raise Exception(f"WARNING: Could not save attribute {attr_key} with value {attr_val} for asset {key}")

            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val


def get_weights_path(encoder_type, encoder_name):
    """
    Retrieve the path to the weights file for a given model name.

    This function looks up the path to the weights file in a local checkpoint
    registry (local_ckpts.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory.

    Args:
        weights_root (str): The root directory where weights files are stored.
        name (str): The name of the model whose weights path is to be retrieved.

    Returns:
        str: The absolute path to the weights file.
    """
    root = os.path.join(os.path.dirname(__file__), f"{encoder_type}_encoder")
    assert encoder_type in [
        "tile",
        "slide",
    ], f"Encoder type must be 'tile' or 'slide', not '{encoder_type}'"
    registry_path = os.path.join(root, "local_ckpts.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)
    path = registry.get(encoder_name)
    if not path:
        raise ValueError(
            f"Please specify the weights path to '{encoder_name}' in '{registry_path}'"
        )
    path = (
        path
        if os.path.isabs(path)
        else os.path.abspath(os.path.join(root, "model_zoo", path))
    )  # Make path absolute
    if not os.path.exists(path):
        print(
            f"WARNING: Path at '{path}' does not exist. Please double-check the registry in '{registry_path}'"
        )
    return path


def get_model_path(encoder_type, encoder_name):
    """
    Retrieve the path to the model file for a given model name.
    Args:
        weights_root (str): The root directory where weights files are stored.
        name (str): The name of the model whose weights path is to be retrieved.

    Returns:
        str: The absolute path to the weights file.
    """
    root = os.path.join(os.path.dirname(__file__), f"{encoder_type}_encoder")
    assert encoder_type in [
        "tile",
        "slide",
    ], f"Encoder type must be 'tile' or 'slide', not '{encoder_type}'"
    registry_path = os.path.join(root, "local_ckpts.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)
    path = registry.get(encoder_name)
    if not path:
        raise ValueError(
            f"Please specify the model path to '{encoder_name}' in '{registry_path}'"
        )
    path = (
        path
        if os.path.isabs(path)
        else os.path.abspath(os.path.join(root, "model_zoo", path))
    )  # Make path absolute
    if not os.path.exists(path):
        print(
            f"WARNING: Path at '{path}' does not exist. Please double-check the registry in '{registry_path}'"
        )
    return path
