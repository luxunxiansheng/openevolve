# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


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
    n = 26
    centers = np.zeros((n, 2))

    # Place a central circle
    centers[0] = [0.5, 0.5]

    # Helper function to place circles in a ring
    def place_ring(center, radius, num_circles, start_index):
        for i in range(num_circles):
            angle = 2 * np.pi * i / num_circles
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            centers[start_index + i] = [x, y]

    # Place 8 circles in an inner ring
    place_ring([0.5, 0.5], 0.3, 8, 1)

    # Place 16 circles in an outer ring
    place_ring([0.5, 0.5], 0.7, 16, 9)

    # Ensure all centers are within the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


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
    radii = np.ones(n)

    # Compute distances to square borders (vectorized)
    border_distances = np.minimum(centers[:, 0], centers[:, 1])
    border_distances = np.minimum(border_distances, 1 - centers[:, 0])
    border_distances = np.minimum(border_distances, 1 - centers[:, 1])
    radii = border_distances.copy()

    # Limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END