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


def plot_2Dproj_img_grid(emb_2d, images, savepath=None, grid_size=50, img_size=32):
    """
    Display images on a grid based on t-SNE coordinates (no overlap).

    emb_2d : (N,2) array of t-SNE coordinates
    images : list/array of images (H,W,C)
    grid_size : size of the grid (grid_size x grid_size)
    img_size : final size of each thumbnail in pixels
    """
    # Normalize coordinates to [0, 1]
    x = (emb_2d[:, 0] - emb_2d[:, 0].min()) / (emb_2d[:, 0].max() - emb_2d[:, 0].min())
    y = (emb_2d[:, 1] - emb_2d[:, 1].min()) / (emb_2d[:, 1].max() - emb_2d[:, 1].min())

    # Grid indices
    gx = np.floor(x * (grid_size - 1)).astype(int)
    gy = np.floor(y * (grid_size - 1)).astype(int)

    # Final canvas (empty grid)
    canvas = 255 * np.ones(
        (grid_size * img_size, grid_size * img_size, 3), dtype=np.uint8
    )

    filled = set()
    for i in range(len(images)):
        cell = (gx[i], gy[i])
        if cell in filled:  # if already filled, skip
            continue
        filled.add(cell)

        # Crop and resize the image
        img = images[i]
        if img.shape[0] != img_size or img.shape[1] != img_size:
            from skimage.transform import resize

            img = (255 * resize(img, (img_size, img_size), anti_aliasing=True)).astype(
                np.uint8
            )

        # Position in the grid
        x0, y0 = cell[0] * img_size, cell[1] * img_size
        canvas[y0 : y0 + img_size, x0 : x0 + img_size] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
