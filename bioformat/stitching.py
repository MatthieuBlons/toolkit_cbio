import pyvips
import matplotlib.pyplot as plt
import tifffile
import matplotlib.patches as patches
import random
from xml.etree import ElementTree as ET
from bioformat.utils import vips_normalize
from bioformat.write_ome import save_ometif

# TODO homogenize function to only use pyvips (with tiffload)


class region:
    def __init__(
        self,
        XPosition,
        YPosition,
        Height,
        Width,
        filePath,
        XResolution,
        YResolution,
        XCord,
        YCord,
    ):
        self.XPosition = XPosition  # Stage position
        self.YPosition = YPosition  # Stage position
        self.Height = Height
        self.Width = Width
        self.filePath = filePath
        self.XResolution = XResolution
        self.YResolution = YResolution
        self.ResolutionUnit = "micron"
        self.XCord = XCord  # pixel position on the stitched image
        self.YCord = YCord  # pixel position on the stitched image


def get_region(filePath):
    with tifffile.TiffFile(filePath) as tif:
        tag_xpos = tif.pages[0].tags["XPosition"]
        tag_ypos = tif.pages[0].tags["YPosition"]
        tag_xres = tif.pages[0].tags["XResolution"]
        tag_yres = tif.pages[0].tags["YResolution"]
        # to investigate
        xpos = 10000 * (tag_xpos.value[0] / tag_xpos.value[1])
        xres = tag_xres.value[0] / (tag_xres.value[1] * 10000)
        ypos = 10000 * tag_ypos.value[0] / tag_ypos.value[1]
        yres = tag_yres.value[0] / (tag_yres.value[1] * 10000)
        height = tif.pages[0].tags["ImageLength"].value
        width = tif.pages[0].tags["ImageWidth"].value
    rg = region(
        xpos * xres,
        ypos * yres,
        height,
        width,
        filePath,
        xres,
        yres,
        xpos * xres,
        ypos * yres,
    )
    return rg


# collect regions
def collect_regions(fileList):
    region_collection = []
    for file in fileList:
        region_collection.append(get_region(file))
    return region_collection


# Calculate image cordinates from the stage cordinates
def zero_center_regions(region_collection):
    minX = min(region.XPosition for region in region_collection)
    minY = min(region.YPosition for region in region_collection)
    for region in region_collection:
        region.XCord = region.XPosition - minX
        region.YCord = region.YPosition - minY
    return region_collection


# count number of channels (n-pages - 1 last page coresponds to the false color thumbnail)
def count_channels(region_collection):
    temp = pyvips.Image.new_from_file(
        region_collection[0].filePath, access="sequential"
    )
    channel_count = int(temp.get("n-pages")) - 1
    return channel_count


def interleaved_tile(filePath, channels):
    tile = pyvips.Image.new_from_file(filePath, n=channels)
    page_height = tile.get("page-height")
    # chop into pages
    pages = [
        tile.crop(0, y, tile.width, page_height)
        for y in range(0, tile.height, page_height)
    ]
    # join pages band-wise to make an interleaved image
    tile = pages[0].bandjoin(pages[1:])
    tile = tile.copy(interpretation="multiband")
    return tile


def rgb_colors(colors):
    new_colors = []
    for cl in colors:
        cl = str.split(cl, ",")
        RGBint = int.from_bytes(
            [int(cl[0]), int(cl[1]), int(cl[2]), 1], byteorder="big", signed=True
        )
        new_colors.append(RGBint)
    return new_colors


# tiff channel meta data to xml
def channel_xml(filePath):
    img = tifffile.TiffFile(filePath)
    imgData_s0 = img.series[0]
    names = [(ET.fromstring(page.description).find("Name").text) for page in imgData_s0]
    colors = [
        (ET.fromstring(page.description).find("Color").text) for page in imgData_s0
    ]
    rgbint = rgb_colors(colors)
    xml = ("\n").join(
        [
            f"""<Channel ID="Channel:0:{i}" Name="{channel_name}" Color="{rgbint[i]}" SamplesPerPixel="1"/>"""
            for i, channel_name in enumerate(names)
        ]
    )
    return xml, names, rgbint


# Plot region locations and bounding box for troubleshooting
def plot_regions(region_collection, show=True):
    rect = []
    cordX = []
    cordY = []
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for region in region_collection:
        rect.append(
            patches.Rectangle(
                (region.XCord, region.YCord),
                region.Width,
                region.Height,
                linewidth=1,
                edgecolor=colors[random.randint(0, 9)],
                facecolor="none",
            )
        )
        cordX.append(region.XCord)
        cordY.append(region.YCord)
    # Create figure and axes
    _, ax = plt.subplots()
    ax.invert_yaxis()
    ax.scatter(cordX, cordY)
    for rect in rect:
        ax.add_patch(rect)
    if show:
        plt.show()


# Get magnification
def get_magnification(filePath):
    img = tifffile.TiffFile(filePath)
    imgData_s0 = img.series[0]
    objective_str = ET.fromstring(imgData_s0[0].description).find("Objective").text
    magnification_str = str.split(objective_str, "x")[0]
    magnification = float(magnification_str)
    return magnification


# Stitch regions and return ome-tiff compatible image
def stitch_regions(region_collection, outFile):
    channel_xmls = []
    channel_names = []
    magnifications = []
    channel_colors = []
    channels = count_channels(region_collection)
    stitched_img = pyvips.Image.black(1, 1, bands=channels)
    for region in region_collection:
        tile = interleaved_tile(region.filePath, stitched_img.bands)
        # insert tile to build final image
        stitched_img = stitched_img.insert(
            tile, region.XCord, region.YCord, expand=1, background=[0]
        )
        xmls, names, colors = channel_xml(region.filePath)
        channel_xmls.append(xmls)
        channel_names.append(names)
        channel_colors.append(colors)
        magnifications.append(get_magnification(region.filePath))

    # TODO
    # check if channels_xml are consistent
    # check if channels_name are consistent
    # check if magnification are consistent

    # normalize stitched_img
    normalized_img, _, _ = vips_normalize(stitched_img)
    normalized_img *= 65535
    # cast to format stitched_img
    final_image = normalized_img.copy()
    final_image = final_image.cast(pyvips.BandFormat.USHORT)
    # save tif
    save_ometif(
        image_pyvips=final_image,
        dst=outFile,
        pixel_size_x=1 / region.XResolution,
        pixel_size_y=1 / region.YResolution,
        magnification=magnifications[0],
        channel_xml=channel_xmls,
        channel_names=channel_names[0],
        channel_colors=channel_colors[0],
        CONFIG_PATH="/Users/mblons/dev/packages/toolkit/bioformat/default_ome_config.xml",
        PYVIPS2OME_FORMAT="/Users/mblons/dev/packages/toolkit/bioformat/PYVIPS2OME_FORMAT.yaml",
    )
