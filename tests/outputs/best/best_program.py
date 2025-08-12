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
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place a central circle
    center_x, center_y = 0.5, 0.5
    centers[0] = [center_x, center_y]

    # Place 8 circles in a ring around the center
    radius = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        centers[i + 1] = [x, y]

    # Place 16 circles in a larger ring
    radius = 0.7
    for i in range(16):
        angle = 2 * np.pi * i / 16
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        centers[i + 9] = [x, y]

    # Ensure all centers are within the square bounds [0.01, 0.99]
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii
    radii = compute_max_radii(centers)

    # Calculate the sum of the radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they do not overlap and are fully contained within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]

    # Initialize all radii to 1 as a baseline value
    radii = np.ones(n)

    # Compute minimum distance from each center to any square border
    border_distances = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1 - centers[:, 0], 1 - centers[:, 1])
    )
    radii = border_distances  # Set each circle's initial radius based on border constraints

    # Adjust for overlap between all pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            distance = np.sqrt(dx**2 + dy**2)
            sum_radius = radii[i] + radii[j]

            if sum_radius > distance:
                scale_factor = distance / (radii[i] + radii[j])
                radii[i] *= scale_factor
                radii[j] *= scale_factor

    return radii


# EVOLVE-BLOCK-END