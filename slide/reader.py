# General libraries
from typing import Optional
import os

# Data libraries
import openslide
import numpy as np
import PIL

mpp_possible_keys = [
    openslide.PROPERTY_NAME_MPP_X,
    "openslide.mirax.MPP",
    "aperio.MPP",
    "hamamatsu.XResolution",
    "openslide.comment",
]


class OpenWSI:
    """Warp around openslide.OpenSlide

    Args:
        img_path (str):
            path to the WSI file.
        name (Optional[str]):
            custom name for the WSI. If None, the filename is used.
        mpp (Optional[float]):
            if known resolution at level 0 in microns per pixel. If None, mpp will be inferred from slide metadata
    """

    def __init__(
        self, img_path: str, name: Optional[str] = None, mpp: Optional[float] = None
    ):
        """Initialize an OpenWSI object from a whole slide image (WSI). Loads the slide using OpenSlide,
        extracts metadata, notably resolution (mpp), and estimates magnification level.

        Args:
            img_path (str):
                Path to the WSI file.
            name (Optional[str]):
                Optional custom name for the WSI. If None, the filename is used.
            mpp (Optional[float]):
                Optional resolution at level 0 in microns per pixel (µm/px).
                If None, it will be automatically inferred from slide metadata.

        Returns:
            None
        """
        self.img_path = img_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(img_path))
        else:
            self.name, self.ext = os.path.splitext(name)
        self.img = openslide.OpenSlide(self.img_path)
        self._fetch_meta()

        self.mpp = mpp
        if self.mpp is None:
            self.mpp = self._fetch_mpp()

        self.magnification = self._fetch_magnification()

    def _fetch_meta(self):
        """Fetch and store slide metadata.

        Args:
            self

        Returns:
            None
        """
        self.dimensions = self.img.dimensions
        self.width, self.height = self.dimensions
        self.level_count = self.img.level_count
        self.level_dimensions = self.img.level_dimensions
        self.level_downsamples = self.img.level_downsamples
        self.properties = self.img.properties

    def _fetch_mpp(self) -> float | None:
        """If necessary, retrieve the resolution (mpp) from slide metadata.

        Args:
            self

        Returns:
            mpp (float or None):
                resolution at level 0
        """

        # Search for mpp_x
        for key in mpp_possible_keys:
            if key in self.img.properties:
                try:
                    mpp_x = float(self.img.properties[key])
                    break
                except ValueError:
                    continue
        # Convert pixel resolution to mpp
        if mpp_x is None:
            x_resolution = self.img.properties.get("tiff.XResolution", None)
            unit = self.img.properties.get("tiff.ResolutionUnit", None)
            if not x_resolution or not unit:
                return None
            if unit == "CENTIMETER" or unit == "centimeter":
                mpp_x = 10000 / float(x_resolution)  # 1 cm = 10,000 microns
            elif unit == "INCH":
                mpp_x = 25400 / float(x_resolution)  # 1 inch = 25,400 microns
            else:
                return None  # Unsupported unit -- add more conditions is needed.
        mpp_x = round(mpp_x, 4)
        return mpp_x

    def _fetch_magnification(self) -> int:
        """Estimate the objective magnification based on resolution at level 0.

        Args:
            self

        Returns:
            mag (int):
                Estimated magnification (e.g., 5x, 10x, 20x, 40x, etc...).

        Raises:
            ValueError:
                If the detected mpp does not correspond to a known
                magnification level.

        """
        if self.mpp is not None:
            mpp_x = self.mpp
        if mpp_x is not None:
            if mpp_x < 0.16:
                return 80
            elif mpp_x < 0.2:
                return 60
            elif mpp_x < 0.3:
                return 40
            elif mpp_x < 0.6:
                return 20
            elif mpp_x < 1.2:
                return 10
            elif mpp_x < 2.4:
                return 5
            else:
                raise ValueError(f"mpp as unexpected value: mpp={mpp_x}")
        else:
            mag = self.img.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            return int(mag)

    def get_thumbnail(self, size: tuple = (1024, 1024)) -> PIL.Image.Image:
        """Generate a thumbnail image of the whole slide.

        Args:
            size (tuple):
                Desired maximum size (width, height) of the thumbnail in pixels.

        Returns:
            PIL.Image.Image:
                Thumbnail image of the WSI.
        """
        return self.img.get_thumbnail(size)

    def read_region(
        self, location: tuple, level: int, size: tuple, numpy: bool = True
    ) -> PIL.Image.Image | np.ndarray:
        """Read a rectangular region from the slide at a specified pyramid level.

        Args:
            location (tuple):
                (x, y) coordinates of the top-left corner at the requested level.
            level (int):
                Pyramid level to read from.
            size (tuple):
                (width, height) of the region to extract.
            numpy (bool):
                If True, returns the region as a NumPy array (RGB).
                If False, returns a PIL image.

        Returns:
            crop (numpy.ndarray or PIL.Image.Image):
                Extracted image region.
        """
        (x0, y0) = self.get_xy_0(location, level, integer=True)
        # location should be a tuple giving the top left pixel in the level 0 reference frame
        crop = self.img.read_region((x0, y0), level, size)
        if numpy:
            crop = np.array(crop)[:, :, :3]
        return crop

    def read_whole(
        self, level: int, numpy: bool = True
    ) -> PIL.Image.Image | np.ndarray:
        """Read the entire slide image at a specified pyramid level.

        Args:
            level (int):
                Pyramid level to read.
            numpy (bool):
                If True, returns the image as a NumPy array (RGB).
                If False, returns a PIL image.

        Returns:
            whole (numpy.ndarray or PIL.Image.Image):
                Whole-slide image at the requested level.
        """
        whole = self.img.read_region(
            location=(0, 0), level=level, size=self.level_dimensions[level]
        )
        if numpy:
            whole = np.array(whole)[:, :, :3]
        return whole

    def get_best_level_for_downsample(
        self, ask_downsample: float, precision: float = 0.01
    ) -> tuple[int, int, float]:
        """Determine the most appropriate pyramid level for a requested
        downsampling factor. If an exact match is not available,
        returns the closest level and a resize factor.

        Args:
            ask_downsample (float):
                Desired downsampling factor relative to level 0.
            precision (float):
                Tolerance to consider a downsample factor as an exact match.

        Returns:
            tuple:
                (level_index, level_downsample, resize_factor)
                where resize_factor indicates any additional scaling needed.

        Raises:
            ValueError:
                If no suitable pyramid level is found.
        """
        level_downsamples = self.level_downsamples
        # First, check for a close match
        for level_best, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - ask_downsample) <= precision:
                return (
                    level_best,
                    level_downsample,
                    1,
                )  # Exact match, no custom downsampling needed
        # If not,
        if ask_downsample >= level_downsamples[0]:
            # Downsampling: find the highest level_downsample less than or equal to the desired downsample
            level_best = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= ask_downsample:
                    level_best = level
                    resize_factor = level_downsample / ask_downsample
                else:
                    break  # level_downsamples are sorted, no need to check further
            if level_best is not None:
                return level_best, level_downsamples[level_best], resize_factor
        else:
            # Upsampling: find the smallest level_downsample greater than or equal to the desired downsample
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= ask_downsample:
                    resize_factor = ask_downsample / level_downsample
                    return level, level_downsamples[level], resize_factor

        # If no suitable level is found, raise an error
        raise ValueError(f"No level found for downsample {ask_downsample}.")

    def get_xy_0(self, point, level, integer=True) -> tuple[int, int]:
        """Convert coordinates from a given pyramid level to level 0 coordinates.

        Args:
            point (tuple):
                (x, y) coordinates at the specified pyramid level.
            level (int):
                Pyramid level of the input point.
            integer (bool):
                If True, returns integer coordinates.
                If False, returns floating-point coordinates.

        Returns:
            point_0 (tuple):
                (x_0, y_0) coordinates in level 0 reference frame.
        """
        x, y = point
        x_0 = x * self.dimensions[0] / self.level_dimensions[level][0]
        y_0 = y * self.dimensions[1] / self.level_dimensions[level][1]
        if integer:
            point_0 = (int(x_0), int(y_0))
        else:
            point_0 = (x_0, y_0)
        return point_0
