# EVOLVE-BLOCK-START
"""Optimized circle packing algorithm for n=26 circles in a unit square"""
import numpy as np


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))

    # Place a central circle at the center of the square
    centers[0] = [0.5, 0.5]

    # Place 8 circles in a ring around the central circle
    place_ring(centers, start_idx=1, num_circles=8, radius=0.25)

    # Place 16 circles in a larger ring around the central circle
    place_ring(centers, start_idx=9, num_circles=16, radius=0.4)

    # Ensure all centers are within the unit square bounds
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for each circle position
    radii = compute_max_radii(centers)

    return centers, radii, np.sum(radii)


def place_ring(centers, start_idx, num_circles, radius, center_x=0.5, center_y=0.5):
    """
    Places a ring of circles around a given center point.

    Args:
        centers: numpy array of circle centers
        start_idx: index to start placing circles
        num_circles: number of circles in the ring
        radius: radius of the ring (distance from center)
        center_x, center_y: coordinates of the central point
    """
    for i in range(num_circles):
        angle = 2 * np.pi * i / num_circles
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        centers[start_idx + i] = [x, y]
    return centers


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]

    # Step 1: Compute initial radii based on distance to square boundaries
    horizontal_distance = np.minimum(centers[:, 0], 1 - centers[:, 0])
    vertical_distance = np.minimum(centers[:, 1], 1 - centers[:, 1])
    radii = np.minimum(horizontal_distance, vertical_distance)

    # Step 2: Adjust radii to prevent overlaps between circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            distance = np.hypot(dx, dy)
            if radii[i] + radii[j] > distance:
                scale = distance / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
# EVOLVE-BLOCK-END