import numpy as np
from scipy.ndimage import gaussian_filter1d


def dynamic_display_range(image, smoothing_sigma=1.0, min_density_ratio=0.001):
    """
    Compute the dynamic display range for an image channel using histogram analysis.

    Parameters:
        image (ndarray): Input image channel as a NumPy array.
        smoothing_sigma (float): Sigma for Gaussian smoothing of the histogram.
        min_density_ratio (float): Minimum histogram density (relative to peak) to include in the range.

    Returns:
        tuple: (lower_bound, upper_bound) for the display range.
    """
    # Compute the histogram
    hist, bin_edges = np.histogram(
        image, bins=256, range=(np.min(image), np.max(image)), density=True
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth the histogram to reduce noise
    smoothed_hist = gaussian_filter1d(hist, sigma=smoothing_sigma)
    # Determine the peak density
    peak_density = np.max(smoothed_hist)

    # Find the bounds where the histogram density exceeds a threshold
    threshold = peak_density * min_density_ratio
    valid_bins = bin_centers[smoothed_hist > threshold]

    # Set the lower and upper bounds
    lower_bound = valid_bins[0] if len(valid_bins) > 0 else np.min(image)
    upper_bound = valid_bins[-1] if len(valid_bins) > 0 else np.max(image)

    return lower_bound, upper_bound


def percentil_range(image, p_low=1, p_high=99):
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)
    return lo, hi


def blend_colors(
    img,
    colors,
    scale_by="hist",
    density_ratio=0.0001,
    gammas: float | list[float] = 1,
    alpha: float | list[float] = 1,
    blending="add",
    eps=1e-8,
):

    if len(colors.shape) > 1:
        n_channel_color = colors.shape[1]
    else:
        n_channel_color = len(colors)
        colors = np.expand_dims(colors, axis=0)

    if img.ndim > 2:
        r, c, nc = img.shape[:3]
    else:
        nc = 1
        r, c = img.shape[:2]
        img = np.expand_dims(img, axis=-1)

    relative_imgs = []
    for i in range(nc):
        channel = img[..., i]
        # relative image is how bright the channel will be
        if scale_by == "hist":
            lower_bound, upper_bound = dynamic_display_range(
                channel, min_density_ratio=density_ratio
            )
            channel = np.clip(channel, a_min=lower_bound, a_max=upper_bound)
        elif scale_by == "percentile":
            lower_bound, upper_bound = percentil_range(channel)
            channel = np.clip(channel, a_min=lower_bound, a_max=upper_bound)
        relative_img = (channel - channel.min()) / (channel.max() - channel.min() + eps)

        if isinstance(gammas, list):
            assert (
                len(gammas) == nc
            ), "if you want per channel gammas, provide a list that matches"
            relative_img = np.power(relative_img, gammas[i])
        else:
            relative_img = np.power(relative_img, gammas)
        relative_imgs.append(relative_img)
    relative_imgs = np.stack(relative_imgs, axis=-1)

    # blending (switch between method)
    blended_img = np.zeros((r, c, n_channel_color))
    if blending == "add":
        # Additive mixing
        for i in range(nc):
            for j in range(n_channel_color):
                channel_color = colors[i, j]
                blended_img[..., j] += channel_color * relative_imgs[..., i]
        blended_img = np.clip(blended_img, a_min=0, a_max=255)
        blended_img = blended_img.astype(np.uint8)

    elif blending == "max":
        argmax = np.argmax(relative_imgs, axis=-1)
        blended_img = np.zeros((r, c, n_channel_color))
        for i in range(nc):
            mask = argmax == i
            for j in range(n_channel_color):
                channel_color = colors[i, j]
                blended_img[..., j][mask] = channel_color * relative_imgs[..., i][mask]
        blended_img = blended_img.astype(np.uint8)

    # Add Max projection style

    return blended_img
