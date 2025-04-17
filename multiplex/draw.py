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
    smoothed_hist = hist
    # Determine the peak density
    peak_density = np.max(smoothed_hist)

    # Find the bounds where the histogram density exceeds a threshold
    threshold = peak_density * min_density_ratio
    valid_bins = bin_centers[smoothed_hist > threshold]

    # Set the lower and upper bounds
    lower_bound = valid_bins[0] if len(valid_bins) > 0 else np.min(image)
    upper_bound = valid_bins[-1] if len(valid_bins) > 0 else np.max(image)

    return lower_bound, upper_bound


def blend_colors(img, colors, scale_by="clip", density_ratio=0.0001, gamma=1):
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

    blended_img = np.zeros((r, c, n_channel_color))
    for i in range(nc):
        channel = img[..., i]
        # relative image is how bright the channel will be
        if scale_by != "clip":
            relative_img = (channel - channel.min()) / (channel.max() - channel.min())
        else:
            lower_bound, upper_bound = dynamic_display_range(
                channel, min_density_ratio=density_ratio
            )
            clipped = np.clip(channel, a_min=lower_bound, a_max=upper_bound)
            relative_img = (clipped - clipped.min()) / (clipped.max() - clipped.min())

        # blending is how to weight the mix of colors, similar to an alpha channel
        for j in range(n_channel_color):
            channel_color = colors[i, j]
            blended_img[..., j] += (channel_color / 255.0) * relative_img
    blended_img = np.clip(blended_img, a_min=0, a_max=1)
    blended_img = np.power(blended_img, gamma)
    blended_img = (blended_img * 255).astype(np.uint8)
    return blended_img
