from typez import NDArray
from kmeans_image import compute_image_kmeans, sort_colors_by_value
from katz.imaqt import IMAQT
import logging
import sys
import numpy as np

log = logging.getLogger(__name__)

# Set palette of superPiano to means of image.

# Emits data of form:
"""
{"type":"colors","value":[[1,1,1],[29,12,4],[4,10,31],[44,32,24],[21,33,53],[51,47,60],[55,82,122],[108,113,141],[229,130,81],[239,115,135],[250,245,242],[253,209,155]]}
"""

PARAMETER_TOPIC = "orgb-param"

if __name__ == "__main__":
    log.info(f"Evaluating means of {sys.argv[1]}")
    colors: NDArray[["N", 3], np.uint8] = compute_image_kmeans(
        sys.argv[1], n_clusters=12
    ) 
    log.info(f"Centers: {colors}")

    hue_sorted_colors = sort_colors_by_value(colors)
    log.info(f"HSV Centers, sorted: {hue_sorted_colors}")

    IMAQT.single(PARAMETER_TOPIC, {"type": "colors", "value": hue_sorted_colors.tolist()})