import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from operator import itemgetter


def mosaic(imlist, size=None, max_img=None, ax=None):
    if not imlist:
        raise Exception("List of images is empty")
    if not size:
        imsizes = []
        if isinstance(imlist[0], str):
            for name in imlist:
                with Image.open(name) as img:
                    imsizes.append(img.size)
            max_width, max_height = (
                max(imsizes, key=itemgetter(0))[0],
                max(imsizes, key=itemgetter(1))[1],
            )
        size = (max_width, max_height)
    if not max_img:
        max_img = len(imlist)
    if not ax:
        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, fig=fig)
    ax.set_axis_off()
    grid_size = (
        np.ceil(np.sqrt(max_img)).astype(int),
        np.ceil(np.sqrt(max_img)).astype(int),
        3,
    )
    grid = np.zeros(
        (grid_size[0] * size[0], grid_size[1] * size[1], grid_size[2])
    ).astype(int)
    # populate the mosaic
    gridpos = 0
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if gridpos < max_img:
                # make it possible to use list of numpy as the image list
                if isinstance(imlist[0], str):
                    with Image.open(imlist[gridpos]) as img:
                        if img.size != size:
                            img_resized = img.resize(size)
                            np_img = np.array(img_resized.convert(mode="RGB"))
                        else:
                            np_img = np.array(img.convert(mode="RGB"))
                        grid[
                            x * size[0] : (x + 1) * size[0],
                            y * size[1] : (y + 1) * size[1],
                            :,
                        ] = np_img[:, :, :]
                        gridpos += 1
                elif isinstance(imlist[0], np.ndarray):
                    np_img = imlist[gridpos]
                    grid[
                        x * size[0] : (x + 1) * size[0],
                        y * size[1] : (y + 1) * size[1],
                        :,
                    ] = np_img[:, :, :]
                    gridpos += 1
                else:
                    raise TypeError(
                        "imlist should be a list of str or list of ndarrays"
                    )
    # show image
    ax.imshow(grid)


def pointillism(colorlist, dotsize=(16, 16), ax=None):
    nb_dots = len(colorlist)
    if not ax:
        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, fig=fig)
    ax.set_axis_off()
    grid_size = (
        np.ceil(np.sqrt(nb_dots)).astype(int),
        np.ceil(np.sqrt(nb_dots)).astype(int),
        3,
    )
    grid = np.zeros(
        (grid_size[0] * dotsize[0], grid_size[1] * dotsize[1], grid_size[2])
    ).astype(int)
    # populate the mosaic
    gridpos = 0
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if gridpos < nb_dots:
                grid[
                    x * dotsize[0] : (x + 1) * dotsize[0],
                    y * dotsize[1] : (y + 1) * dotsize[1],
                    :,
                ] = colorlist[gridpos]
                gridpos += 1
    # show image
    ax.imshow(grid)
