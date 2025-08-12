# EVOLVE-BLOCK-START
"""Configurable circle packing with optimized parameters and improved maintainability"""

import numpy as np


# Configuration parameters for circle arrangement
NUM_CENTRAL = 1
NUM_INNER_RING = 8
NUM_OUTER_RING = 17
n = NUM_CENTRAL + NUM_INNER_RING + NUM_OUTER_RING
assert n == 26, "Total circles must be 26"

# Optimization parameter for convergence iterations
MAX_ITERATIONS = 10


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize centers array
    centers = np.zeros((n, 2))

    # Place the central circle
    centers[0] = [0.5, 0.5]

    # Place inner ring circles
    inner_angles = np.linspace(0, 2 * np.pi, NUM_INNER_RING, endpoint=False)
    inner_positions = 0.5 + 0.3 * np.array([np.cos(inner_angles), np.sin(inner_angles)]).T
    centers[1:1 + NUM_INNER_RING] = inner_positions

    # Place outer ring circles
    outer_angles = np.linspace(0, 2 * np.pi, NUM_OUTER_RING, endpoint=False)
    outer_positions = 0.5 + 0.7 * np.array([np.cos(outer_angles), np.sin(outer_angles)]).T
    centers[1 + NUM_INNER_RING:1 + NUM_INNER_RING + NUM_OUTER_RING] = outer_positions

    # Ensure all circles are within the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they do not overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Step 1: Limit radii by distance to square borders
    x = centers[:, 0]
    y = centers[:, 1]
    radii = np.minimum(x, np.minimum(y, np.minimum(1 - x, 1 - y)))

    # Step 2: Iteratively adjust radii to prevent overlaps
    for _ in range(MAX_ITERATIONS):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale

    return radii
# EVOLVE-BLOCK-END