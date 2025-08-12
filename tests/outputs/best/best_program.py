# EVOLVE-BLOCK-START
"""
Configurable circle packing with optimized parameters and improved maintainability

This implementation provides a flexible and efficient approach to circle packing in a unit square.
The configuration is modular, and the core algorithm avoids O(n²) nested loops by using vectorized
operations for pairwise distance computation.
"""

import numpy as np


def construct_packing(num_central=1, num_inner_ring=8, num_outer_ring=17):
    """
    Construct a specific arrangement of circles in a unit square
    that attempts to maximize the sum of their radii.

    Args:
        num_central: Number of central circles.
        num_inner_ring: Number of inner ring circles.
        num_outer_ring: Number of outer ring circles.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = num_central + num_inner_ring + num_outer_ring
    assert n == 26, "Total circles must be 26"

    # Initialize centers array
    centers = np.zeros((n, 2))

    # Place the central circle
    centers[0] = [0.5, 0.5]

    # Place inner ring circles
    inner_angles = np.linspace(0, 2 * np.pi, num_inner_ring, endpoint=False)
    inner_positions = 0.5 + 0.3 * np.array([np.cos(inner_angles), np.sin(inner_angles)]).T
    centers[1:1 + num_inner_ring] = inner_positions

    # Place outer ring circles
    outer_angles = np.linspace(0, 2 * np.pi, num_outer_ring, endpoint=False)
    outer_positions = 0.5 + 0.7 * np.array([np.cos(outer_angles), np.sin(outer_angles)]).T
    centers[1 + num_inner_ring:1 + num_inner_ring + num_outer_ring] = outer_positions

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

    # Step 2: Compute minimum pairwise distances using vectorized operations
    # Compute all pairwise distances
    distances = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

    # Create a mask to exclude self distances (diagonal elements)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)

    # Compute the minimum distance for each circle
    min_dist = np.min(distances[mask].reshape(n, -1), axis=1)

    # Step 3: Set radii as the minimum of border distance and half of min_dist
    radii = np.minimum(radii, min_dist / 2)

    return radii
# EVOLVE-BLOCK-END