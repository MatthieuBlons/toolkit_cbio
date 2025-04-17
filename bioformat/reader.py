import pyvips
from typing import Optional
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from multiplex.draw import blend_colors
from cv2 import convertScaleAbs


# could add the possibility to patch array directly the the OpenOME class
# OpenOME class should be consistent with OpenSlide
class OpenOME:
    """
    The `OpenOME` class provides an interface to work with Bioformat images using pyvips.
    It mirors the basic operation of OpenSlide class objects provided by the OpenSlide project.

    Attributes:
    -----------
    img_path : str
        Path to the WSI file.

    """

    def __init__(
        self, img_path: str, name: Optional[str] = None, mpp: Optional[float] = None
    ):
        """
        Initialize the `OpenSlideWSI` object for working with a Whole Slide Image (WSI).

        Args:
        -----

        Example:
        --------
        """
        self.img_path = img_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(img_path))
        else:
            self.name, self.ext = os.path.splitext(name)
        self.img = pyvips.Image.new_from_file(self.img_path, access="sequential")
        self.mpp = mpp
        self._fetch_meta()
        self.dimensions = (self.width, self.height)
        self.mag = None
        self._fetch_magnification()
        self._fetch_level_count()
        self.level_dimensions = self._fetch_level_dimensions()
        self.level_downsamples = self._fetch_level_downsamples()
        self.check_if_rgb()
        self.check_if_multiplex()

    def description(self, verbose=False):
        description = self.img.get("image-description")
        if verbose:
            print(description)
        return description

    def fields(self, verbose=False):
        fields = self.img.get_fields()
        if verbose:
            print(fields)
        return fields

    def _fetch_meta(self):
        fields = self.fields()
        pixel_infos = self.get_pixel_info()
        if "vips-loader" in fields:
            self.loader = self.img.get("vips-loader")
        if "format" in fields:
            self.format = self.img.get("format")
        if "width" in fields:
            self.width = self.img.get("width")
        else:
            self.width = pixel_infos["XYZ"][0]
        if "height" in fields:
            self.height = self.img.get("height")
        else:
            self.height = pixel_infos["XYZ"][1]
        self.depth = pixel_infos["XYZ"][2]
        self.channels = pixel_infos["C"]
        self.physical_unit = pixel_infos["physical_unit"]
        self.mpp = round(pixel_infos["physical_size"][0], 4)
        if "n-pages" in fields:
            self.pages = self.img.get("n-pages")
        if "bands" in fields:
            self.bands = self.img.get("bands")
        if self.pages > 1:
            channel_infos = self.get_channel_info()
            self.channel_colors = np.array(channel_infos["falsecolors"])
            self.channel_names = channel_infos["names"]

    def _fetch_magnification(self):
        if self.mpp is not None:
            mpp_x = self.mpp
        else:
            raise ValueError(f"Unable to fetch mpp, must be manually set")
        if mpp_x is not None:
            if mpp_x < 0.16:
                self.magnification = 80
            elif mpp_x < 0.2:
                self.magnification = 60
            elif mpp_x < 0.3:
                self.magnification = 40
            elif mpp_x < 0.6:
                self.magnification = 20
            elif mpp_x < 1.2:
                self.magnification = 10
            elif mpp_x < 2.4:
                self.magnification = 5
            else:
                raise ValueError(f"Identified mpp is too low: mpp={mpp_x}")
        else:
            # Use metadata-based magnification as a fallback if mpp_x is not available
            pixel_infos = self.get_pixel_info()
            self.magnification = pixel_infos["mag"]

    def _fetch_level_count(self):
        fields = self.fields()
        if "n-subifds" in fields:
            self.level_count = self.img.get("n-subifds") + 1
        else:
            self.level_count = 0

    def _fetch_level_dimensions(self):
        level_dimensions = []
        if self.level_count:
            for level in range(self.level_count):
                img_level = pyvips.Image.tiffload(self.img_path, subifd=level - 1)
                dim = (img_level.width, img_level.height)
                level_dimensions.append(dim)
        return level_dimensions

    def _fetch_level_downsamples(self):
        level_downsamples = []
        for level in range(self.level_count):
            down_x, down_y = (
                self.level_dimensions[0][0] / self.level_dimensions[level][0],
                self.level_dimensions[0][1] / self.level_dimensions[level][1],
            )
            down = max(down_x, down_y)
            level_downsamples.append(down)
        return level_downsamples

    # Check if subifs or not
    # get meta data with ometif (use a function from OME.utils)
    def get_pixel_info(self, verbose=False):
        description = self.description()
        pixels = read_ome_xml(description, "Pixels", findall=False)
        info_dict = {
            "XYZ": tuple,
            "C": int,
            "T": int,
            "physical_size": tuple,
            "physical_unit": str,
        }
        X = int(pixels.get("SizeX"))
        Y = int(pixels.get("SizeY"))
        Z = int(pixels.get("SizeZ"))
        C = int(pixels.get("SizeC"))
        T = int(pixels.get("SizeT"))
        info_dict["XYZ"] = (X, Y, Z)
        info_dict["C"] = C
        info_dict["T"] = T
        info_dict["physical_size"] = (
            float(pixels.get("PhysicalSizeX")),
            float(pixels.get("PhysicalSizeY")),
        )
        info_dict["physical_unit"] = pixels.get("PhysicalSizeXUnit")
        # get magnification
        if verbose:
            print(info_dict)
        return info_dict

    # (use a function from OME.utils)
    def get_channel_info(self, verbose=False):
        if not self.pages > 1:
            raise AttributeError("OME with single page has no channels info")
        description = self.description()
        channels = read_ome_xml(description, "Channel", findall=True)
        info_dict = {"num_channels": int, "names": [], "falsecolors": []}
        cnt = 0
        for channel in channels:
            cnt += 1
            color = int(channel.get("Color"))
            # Convert integer color to RGB
            r = (color >> 24) & 0xFF
            g = (color >> 16) & 0xFF
            b = (color >> 8) & 0xFF
            a = color & 0xFF
            info_dict["falsecolors"].append((r, g, b))
            info_dict["names"].append(channel.get("Name"))
        info_dict["num"] = cnt
        if verbose:
            print(
                [
                    f"name: {n} with color: {c}"
                    for (n, c) in zip(info_dict["names"], info_dict["falsecolors"])
                ]
            )
        return info_dict

    def read_whole(self, level, numpy=True):
        if level > self.level_count:
            level = self.level_count
            print("pyramidal level was set to lowest...")
        if self.pages > 1:
            # The full-size image is not in subifd but in the main IFD and can be accessed with subifd=-1.
            pages = [
                pyvips.Image.tiffload(
                    self.img_path, page=i, subifd=level - 1, access="sequential"
                )
                for i in range(0, self.pages)
            ]
            whole = pages[0].bandjoin(pages[1:])
        else:
            whole = pyvips.Image.tiffload(
                self.img_path, subifd=level - 1, access="sequential"
            )
        if numpy:
            whole = whole.numpy()
            if self.format == "ushort":
                whole = convertScaleAbs(whole, alpha=(255.0 / 65535.0))
        return whole

    def read_region(self, location, level, size, numpy=True):
        x, y = location
        if level > self.level_count:
            level = self.level_count
            print("pyramidal level was set to lowest...")
        w, h = size
        dim_0 = self.dimensions
        dim_level = self.level_dimensions
        x, y = get_xy_to(
            (x, y),
            dim_0,
            dim_level[level],
            integer=True,
        )
        img = self.read_whole(level, numpy=False)
        if x + w > img.width or y + h > img.height:
            print(
                f"crop region is out of image bounds {img.width} x {img.height}. size was set accordingly"
            )
            w = img.width - x
            h = img.height - y
        crop = img.crop(x, y, w, h)  # (x, y, w, h)
        if numpy:
            crop = crop.numpy()
            if self.format == "ushort":
                crop = convertScaleAbs(crop, alpha=(255.0 / 65535.0))
        return crop

    def check_if_rgb(self):
        self.is_rgb = False
        if self.pages == 1 and self.bands == 3:
            self.is_rgb = True

    def check_if_multiplex(self):
        self.is_multiplex = False
        if self.pages > 1 and self.bands == 1:
            self.is_multiplex = True

    def get_thumbnail(self, size: tuple = (1024, 1024), numpy=False):
        if self.width > self.height:
            thumbnail_width = size[0]
            thumbnail_height = int(size[1] * self.height / self.width)
        else:
            thumbnail_height = size[1]
            thumbnail_width = int(size[0] * self.width / self.height)
        thumbnail_dimensions = (thumbnail_width, thumbnail_height)
        downsample_factor = max(
            dim / thumb for dim, thumb in zip(self.dimensions, thumbnail_dimensions)
        )
        best_level = self.get_best_level_for_downsample(downsample_factor)
        whole = self.read_whole(level=best_level, numpy=False)

        sacling_final = max(
            dim / thumb
            for dim, thumb in zip(
                self.level_dimensions[best_level], thumbnail_dimensions
            )
        )
        thumbnail = whole.resize(1 / sacling_final)
        thumbnail = thumbnail.numpy()
        if self.format == "ushort":
            thumbnail = convertScaleAbs(
                thumbnail[:, :, : self.pages], alpha=(255.0 / 65535.0)
            )
        if self.is_multiplex:
            thumbnail = blend_colors(thumbnail, self.channel_colors, scale_by="clip")
        if numpy:
            return thumbnail
        else:
            return Image.fromarray(thumbnail)

    def get_single_channel_thumbnail(
        self,
        size: tuple = (1024, 1024),
        key: Optional[str] | None = None,
        idx: int | None = None,
        numpy: bool = False,
    ):
        if not self.is_multiplex:
            raise TypeError(
                "Single channel display is only supported for multiplex images"
            )
        if self.width > self.height:
            thumbnail_width = size[0]
            thumbnail_height = int(size[1] * self.height / self.width)
        else:
            thumbnail_height = size[1]
            thumbnail_width = int(size[0] * self.width / self.height)
        thumbnail_dimensions = (thumbnail_width, thumbnail_height)
        downsample_factor = max(
            dim / thumb for dim, thumb in zip(self.dimensions, thumbnail_dimensions)
        )
        best_level = self.get_best_level_for_downsample(downsample_factor)
        whole = self.read_whole(level=best_level, numpy=False)

        sacling_final = max(
            dim / thumb
            for dim, thumb in zip(
                self.level_dimensions[best_level], thumbnail_dimensions
            )
        )
        thumbnail = whole.resize(1 / sacling_final)
        thumbnail = thumbnail.numpy()
        if self.format == "ushort":
            thumbnail = convertScaleAbs(
                thumbnail[:, :, : self.pages], alpha=(255.0 / 65535.0)
            )

        if idx:
            color = self.channel_colors[idx]
            thumbnail = thumbnail[:, :, idx]
            thumbnail = blend_colors(thumbnail, color)

        if key:
            for idx, name in enumerate(self.channel_names):
                if key in name:
                    break
            color = self.channel_colors[idx]
            thumbnail = thumbnail[:, :, idx]
            thumbnail = blend_colors(thumbnail, color)

        if not key or idx:
            raise AttributeError("need to provide a channel key or index")

        if numpy:
            return thumbnail
        else:
            return Image.fromarray(thumbnail)

    def get_best_level_for_downsample(self, downsample: float, tolerance: float = 0.01):
        # First, check for an exact match within tolerance
        level_downsamples = self.level_downsamples

        for level, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - downsample) <= tolerance:
                return level  # Exact match, no custom downsampling needed

        if downsample >= level_downsamples[0]:
            # Downsampling: find the highest level_downsample less than or equal to the desired downsample
            closest_level = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= downsample:
                    closest_level = level
                else:
                    break  # Since level_downsamples are sorted, no need to check further
            if closest_level is not None:
                return closest_level
        else:
            # Upsampling: find the smallest level_downsample greater than or equal to the desired downsample
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= downsample:
                    return level
        # If no suitable level is found, raise an error
        raise ValueError(f"No suitable level found for downsample {downsample}.")


def get_xy_to(point, dim_from, dim_to, integer=True):
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


def read_ome_xml(xml, key, findall=True):
    root = ET.fromstring(xml)
    namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    search = ".//ome:" + key
    if findall:
        return root.findall(search, namespace)
    else:
        return root.find(search, namespace)
