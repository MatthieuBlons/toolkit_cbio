import pyvips
import numpy as np
from skimage.transform import resize
from xml.etree import ElementTree as ET
import os
import re

# add OPENSLIDE FORMATS
VIPS_READABLE_FORMATS = [*pyvips.get_suffixes(), ".ome.tiff", ".ome.tif"]


# Print pyvips infos
def print_vips_info(img):
    # img should be a path or an custom class OpenOME
    if isinstance(img, str):
        path = img
        img = pyvips.Image.new_from_file(img, access="sequential")
    else:
        path = img.image_path
    fields = img.get_fields()
    description = img.get("image-description")
    # Parse XML to get the colors for each channel
    root = ET.fromstring(description)
    namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    print("interpretation: ", img.interpretation)
    print("vips-loader: ", img.get("vips-loader"))
    print("image format: ", img.format)
    print("image coding: ", img.coding)
    print("width: ", img.width)
    print("height: ", img.height)
    print(
        "X resolution: ",
        float(root.find(".//ome:Pixels", namespace).get("PhysicalSizeX")),
    )
    print(
        "Y resolution: ",
        float(root.find(".//ome:Pixels", namespace).get("PhysicalSizeY")),
    )
    print(
        "Resolution unit: ",
        root.find(".//ome:Pixels", namespace).get("PhysicalSizeXUnit"),
    )
    print("X position: ", img.get("xoffset"))
    print("Y position: ", img.get("yoffset"))
    print("pages: ", img.get("n-pages"))
    print("bands: ", img.bands)
    if "n-subifds" in fields:
        print("pyramid levels: ", img.get("n-subifds") + 1)
        print("dimentions level 0 :", (img.width, img.height))
        for level in range(img.get("n-subifds")):
            img_level = pyvips.Image.new_from_file(path, subifd=level)
            print(f"dimentions level {level+1}: ", (img_level.width, img_level.height))


# Normalize a pyvips.Image instance between [0, 1]
def vips_normalize(image_pyvips, percentile=(1, 99)):
    if percentile:
        lower = float(image_pyvips.percent(percentile[0]))
        upper = float(image_pyvips.percent(percentile[1]))
    else:
        lower = image_pyvips.min()
        upper = image_pyvips.max()
    # Clip pixel values using arithmetic operations
    clipped_pyvips = image_pyvips.copy()
    clipped_pyvips = (
        (image_pyvips < lower) / 255 * lower
        + (image_pyvips * (image_pyvips >= lower) / 255 * (image_pyvips <= upper) / 255)
        + (image_pyvips > upper) / 255 * upper
    )
    normalized_pyvips = clipped_pyvips.copy()
    normalized_pyvips = (clipped_pyvips - lower) / (upper - lower)
    return normalized_pyvips, lower, upper


def pyramid_nearest_neighbor(img, downscale, max_layer, preserve_range=True):
    if isinstance(img, str):
        img = pyvips.Image.new_from_file(img, access="sequential")
    pyramid = [img]
    for _ in range(1, max_layer):
        height, width = pyramid[-1].shape[:2]
        next_level = resize(
            pyramid[-1],
            (np.ceil(height / downscale), np.ceil(width / downscale)),
            order=0,  # Nearest-neighbor interpolation
            preserve_range=preserve_range,
            anti_aliasing=False,
        )
        pyramid.append(next_level)
    return pyramid


def check_is_ome(path):
    name, ext = os.path.splitext(os.path.basename(path))
    is_ome = re.search(".ome", name) is not None and re.search(".tif*", ext) is not None
    # Verify that image is valid ome.tiff
    # if is_ome:
    # ome_obj = get_ome_xml(path)
    # if ome_obj is None:
    # is_ome = False

    return is_ome


def get_slide_extension(path):
    _, ext = os.path.splitext(os.path.basename(path))
    return ext


def get_ome_xml(path):
    return "Not implemented yet"


def check_to_use_vips(path):
    ext = get_slide_extension(path)
    use_pyvips = ext.lower() in VIPS_READABLE_FORMATS
    return use_pyvips


def check_flattened_pyramid_tiff(path):
    return "Not implemented yet"


def get_metadata_from_xml(path):
    return "Not implemented yet"


def get_slide_reader(slide):
    return "Not implemented yet"
