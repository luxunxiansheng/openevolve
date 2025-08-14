# EVOLVE-BLOCK-START
"""
Circle Packing for n=26 Circles with Enhanced Efficiency and Clarity

This version leverages vectorized operations to eliminate nested loops in radius computation,
restructures the configuration logic for improved readability and modularity, and ensures
a cleaner separation of boundary constraint and overlap checks for better performance.
"""

import numpy as np


def construct_packing():
    """
    Constructs an optimal arrangement of 26 circles in a unit square to maximize sum of radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii):
        - centers: np.ndarray of shape (26, 2), containing circle centers
        - radii: np.ndarray of shape (26,), containing maximized radii without overlaps
        - sum_of_radii: float, sum of all computed radii
    """
    # Configuration of circle arrangement
    num_circles = 26
    central_offset = 0.5
    central_coords = (central_offset, central_offset)
    first_ring_size = 8
    first_ring_radius = 0.3
    outer_ring_size = 17
    outer_ring_radius = 0.7
    boundary_clip = (0.01, 0.99)  # Boundary buffer for safe wall distances

    centers = np.zeros((num_circles, 2))

    # Step 1: Define the central circle
    centers[0] = central_coords

    # Step 2: Place rings around the central circle
    place_ring(
        centers,
        base_index=1,
        num_circles=first_ring_size,
        center=central_coords,
        radius=first_ring_radius,
    )
    place_ring(
        centers,
        base_index=1 + first_ring_size,
        num_circles=outer_ring_size,
        center=central_coords,
        radius=outer_ring_radius,
    )

    # Step 3: Ensure no circle reaches the square boundary by clipping
    np.clip(centers, boundary_clip[0], boundary_clip[1], out=centers)

    # Step 4: Calculate maximal radii using vectorized pairwise distance analysis
    radii = compute_max_radii(centers)

    sum_of_radii = np.sum(radii)

    return centers, radii, sum_of_radii


def place_ring(centers, base_index, num_circles, center, radius):
    """
    Positions a ring of circles equidistantly around a central location.

    Args:
        centers: np.ndarray of shape (n, 2)
        base_index: int, starting position for new ring
        num_circles: int, number of circles in the ring
        center: tuple (x, y), center of circle distribution
        radius: float, radial distance from center to each circle
    """
    angles = np.linspace(0, 2 * np.pi, num_circles, endpoint=False)
    # Use vector operations for improved readability and efficiency
    cx, cy = center
    circle_x = cx + radius * np.cos(angles)
    circle_y = cy + radius * np.sin(angles)
    positions = np.column_stack((circle_x, circle_y))
    centers[base_index : base_index + num_circles] = positions


def compute_max_radii(centers):
    """
    Vectorized algorithm for determining maximum allowable radii:
    - Radii determined by distance to walls
    - Radii constrained to maintain non-overlapping

    Args:
        centers: np.ndarray of shape (n, 2)

    Returns:
        np.ndarray of shape (n), with maximum non-overlapping radii
    """
    n = centers.shape[0]
    radii = np.zeros(n)

    # Compute Euclidean pairwise distances using vector math
    center_differences = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(center_differences**2, axis=2))

    # Remove self-distance from consideration by assigning ∞ to diagonal
    distance_matrix[range(n), range(n)] = np.inf
    min_pairwise_distances = np.min(distance_matrix, axis=1)

    # Determine distance to walls
    x, y = centers[:, 0], centers[:, 1]
    wall_constraints = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))

    # Use pairwise distance for overlap prevention, wall constraints for boundary
    radii = np.minimum(wall_constraints, min_pairwise_distances / 2.0)

    return radii


# EVOLVE-BLOCK-END
