import pathlib
import sys

from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from typez import NDArray
import typez
import pathlib

DOWNSCALE_FACTOR = 4

import logging

log = logging.getLogger(__name__)


def compute_image_kmeans(
    path: str, n_clusters: int, hsvspace: bool = True
) -> NDArray[["N", 3], np.uint8]:
    """
    Compute chromatic kmeans of an image.

    :path: Path to image.
    :n_clusters: Number of clusters.
    :hsvspace: Whether to use HSV space.
    :return: Chromatic kmeans of image.
    """
    image = Image.open(path)
    image_resized = image.resize(
        (image.size[0] // DOWNSCALE_FACTOR, image.size[1] // DOWNSCALE_FACTOR),
        Image.LANCZOS,
    )

    if hsvspace:
        data = np.asarray(image_resized)
        data_vector: NDArray[["N", 3], np.uint8] = data.reshape(
            (data.shape[0] * data.shape[1], data.shape[2])
        )
        hsv_data_vector: NDArray[["N", 3], np.uint8] = matplotlib.colors.rgb_to_hsv(
            data_vector
        )

        # hsv_data_vector is ([0,1], [0,1], [0,255]) data

        # Normalize v value from [0,255] to [0,1] (otherwise L2 distance will favor distance in v-dimension)
        hsv_data_vector_normalized = np.apply_along_axis(
            lambda pixel: (pixel[0], pixel[1], pixel[2] / 255.0), 1, hsv_data_vector
        )

        assert np.all(
            (0.0 <= hsv_data_vector_normalized.reshape(-1))
            & (hsv_data_vector_normalized.reshape(-1) <= 1.0)
        )

        km = KMeans(n_clusters=n_clusters, verbose=True)
        log.info("Fitting...")
        km.fit(hsv_data_vector_normalized)
        log.info("Fitted")

        centers_denormalized = np.apply_along_axis(
            lambda pixel: (pixel[0], pixel[1], pixel[2] * 255.0), 1, km.cluster_centers_
        )
        rgb: NDArray[["N", 3], np.uint8] = matplotlib.colors.hsv_to_rgb(
            centers_denormalized
        )
        return rgb.astype(np.uint8)

    else:
        data = np.asarray(image_resized)
        data_vector: NDArray[["N", 3], np.uint8] = data.reshape(
            (data.shape[0] * data.shape[1], data.shape[2])
        )

        km = KMeans(n_clusters=n_clusters, verbose=True)
        log.info("Fitting...")
        km.fit(data_vector)
        log.info("Fitted")

        return km.cluster_centers_.astype(np.uint8)


def sort_colors_by_value(
    colors: typez.NDArray[["N", 3], np.uint8]
) -> typez.NDArray[["N", 3], np.uint8]:
    return np.array(sorted(colors, key=lambda x: matplotlib.colors.rgb_to_hsv(x)[2]))


def plot_colors(colors: typez.NDArray[["N", 3], np.uint8]) -> None:
    colors_image: typez.NDArray[["N", 3], np.uint8] = colors.reshape(
        (1, colors.shape[0], colors.shape[1])
    )
    plt.imshow(colors_image)
    plt.show()
    return


def save_colors_as_image(colors: typez.NDArray[["N", 3], np.uint8], path: str) -> None:
    colors_2d = colors.reshape((1, colors.shape[0], colors.shape[1]))
    image = Image.fromarray(colors_2d).resize((colors.shape[0] * 72, colors.shape[1] * 72), resample=Image.NEAREST)
    image.save(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    log.info(f"Evaluating means of {sys.argv[1]}")
    colors: NDArray[["N", 3], np.uint8] = compute_image_kmeans(
        sys.argv[1], n_clusters=12
    ) 
    log.info(f"Centers: {colors}")

    hue_sorted_colors = sort_colors_by_value(colors)
    log.info(f"HSV Centers, sorted: {hue_sorted_colors}")

    if len(sys.argv) > 2:
        save_colors_as_image(hue_sorted_colors, sys.argv[2])
    else:
        plot_colors(hue_sorted_colors)
