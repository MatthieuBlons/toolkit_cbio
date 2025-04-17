from typing import Optional
import os
import openslide
import numpy as np


# could add the possibility to patch array directly the the OpenOME class
# OpenOME class should be consistent with OpenSlide
class OpenWSI:
    """
    Attributes:
    -----------
    img_path : str
        Path to the WSI file.

    """

    def __init__(
        self, img_path: str, name: Optional[str] = None, mpp: Optional[float] = None
    ):
        """

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
        self.img = openslide.OpenSlide(self.img_path)
        self._fetch_meta()
        self.width, self.height = self.dimensions
        self.mpp = mpp
        if not mpp:
            self.mpp = self._fetch_mpp()
        self.magnification = self._fetch_magnification()

    def _fetch_meta(self):
        self.dimensions = self.img.dimensions
        self.level_count = self.img.level_count
        self.level_dimensions = self.img.level_dimensions
        self.level_downsamples = self.img.level_downsamples
        self.properties = self.img.properties

    def _fetch_mpp(self):
        openslide_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            "openslide.mirax.MPP",
            "aperio.MPP",
            "hamamatsu.XResolution",
            "openslide.comment",
        ]
        # Search for mpp_x
        for key in openslide_keys:
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

    def _fetch_magnification(self):
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
                raise ValueError(f"Identified mpp is too low: mpp={mpp_x}")
        else:
            mag = self.img.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            return int(mag)

    def get_thumbnail(self, size: tuple = (1024, 1024)):
        return self.img.get_thumbnail(size)

    def get_thumbnail(self, size: tuple = (1024, 1024)):
        return self.img.get_thumbnail(size)

    def read_region(self, location: tuple, level: int, size: tuple, numpy: bool = True):
        crop = self.img.read_region(location, level, size)
        if numpy:
            crop = np.array(crop)[:, :, :3]
        return crop

    def read_whole(self, level: int, numpy: bool = True):
        whole = self.img.read_region(
            location=(0, 0), level=level, size=self.level_dimensions(level)
        )
        if numpy:
            whole = np.array(whole)[:, :, :3]
        return whole

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
        raise ValueError(f"No level found for downsample {downsample}.")
