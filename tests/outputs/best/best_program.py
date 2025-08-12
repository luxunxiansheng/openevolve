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

    # Central circle
    centers[0] = [0.5, 0.5]

    # Function to place a ring of circles
    def _place_ring(start_index, center_x, center_y, radius, num_circles):
        for i in range(num_circles):
            angle = 2 * np.pi * i / num_circles
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            centers[start_index + i] = [x, y]

    # First ring: 8 circles
    _place_ring(1, 0.5, 0.5, 0.3, 8)

    # Second ring: 16 circles
    _place_ring(9, 0.5, 0.5, 0.7, 16)

    # Ensure all circles fit within the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii
    radii = compute_max_radii(centers)

    # Calculate sum of radii
    sum_of_radii = np.sum(radii)

    return centers, radii, sum_of_radii


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

    # Limit radii to maintain margin for square boundaries
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Avoid overlaps by adjusting pair-wise sum of radii to their center distance
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = centers[i] - centers[j]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if radii[i] + radii[j] > distance:
                scale = distance / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END