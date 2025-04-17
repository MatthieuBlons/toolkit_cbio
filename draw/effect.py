import numpy as np
from skimage.transform import resize


def blend_images(im1, im2, background_color=(0, 0, 0), alpha=0.5):
    """
    Blend two RGB images, ignoring the background of the second image if needed.
    Returns:
    - np.ndarray: Blended RGB image.
    """
    # TODO assert both rgb
    # assert same shape
    shape = im1.shape
    if im2.shape != shape:
        im2 = resize(im2, shape, preserve_range=True, order=0)
    # Create a mask where im2 is not background
    mask = np.any(im2 != np.array(background_color), axis=-1)
    # Blend the images
    blended_image = im1.copy()
    for c in range(3):  # Iterate over RGB channels
        blended_image[:, :, c] = np.where(
            mask, (alpha) * im2[:, :, c] + (1 - alpha) * im1[..., c], im1[..., c]
        )
    # TODO add the possibility to Scale back to 0-255 range and convert to uint8 (or other dtype) if needed
    return blended_image
