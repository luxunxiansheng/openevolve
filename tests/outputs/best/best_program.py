# EVOLVE-BLOCK-START
"""
Optimized and well-documented circle-packing algorithm for 26 circles in a unit square
Maximizing the sum of radii with improved clarity, maintainability, and correctness.
"""

import numpy as np
from collections import defaultdict


def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square
    that maximizes the sum of their radii, ensuring no overlaps.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        - centers: np.array of shape (26, 2) with (x, y) coordinates
        - radii: np.array of shape (26) with radius of each circle
        - sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))

    # Set the central circle
    centers[0] = [0.5, 0.5]

    def _place_circular_ring(start_index, center, ring_radius, num_circles):
        """
        Place a uniform ring of circles around a given coordinate.

        Args:
            start_index: First index to assign positions in centers array
            center: (x, y) position of the ring center
            ring_radius: Radius of the ring (from center to any circle on it)
            num_circles: Number of circles in the ring
        """
        cx, cy = center
        for idx in range(num_circles):
            angle = 2 * np.pi * idx / num_circles
            x = cx + ring_radius * np.cos(angle)
            y = cy + ring_radius * np.sin(angle)
            centers[start_index + idx] = [x, y]

    # Place first ring of 8 circles
    _place_circular_ring(start_index=1, center=(0.5, 0.5), ring_radius=0.3, num_circles=8)

    # Place second ring of 16 circles
    _place_circular_ring(start_index=9, center=(0.5, 0.5), ring_radius=0.7, num_circles=16)

    # Ensure all circles stay within the 1% boundary margin
    centers = np.clip(centers, 0.01, 0.99)

    # Compute optimal radii by finding the minimum distance to other circles
    radii = compute_optimal_radii(centers)

    return centers, radii, np.sum(radii)


def compute_optimal_radii(centers):
    """
    Calculate the optimal radii for all circles so that:
    - No two circles overlap
    - Radii are limited by distance to edge of square
    Uses grid partitioning for efficiency and correctness.

    Args:
        centers: np.array of (n, 2) circle positions

    Returns:
        np.array of valid radii for all circles
    """
    n = centers.shape[0]

    # Set initial radii to the distance to nearest square boundary
    edge_distances = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1 - centers[:, 0], 1 - centers[:, 1])
    )
    radii = edge_distances.copy()

    # Use grid spatial partitioning to reduce comparisons
    GRID_CELL_SIZE = 0.2  # Split square into 5x5 grid
    grid_map = defaultdict(list)

    # Map each circle to its grid cell
    for idx, (x, y) in enumerate(centers):
        cell_x = int(x / GRID_CELL_SIZE)
        cell_y = int(y / GRID_CELL_SIZE)
        grid_map[(cell_x, cell_y)].append(idx)

    # Find the closest other circle for each circle
    closest_distances = [np.inf] * n
    for idx, (x, y) in enumerate(centers):
        cell_x = int(x / GRID_CELL_SIZE)
        cell_y = int(y / GRID_CELL_SIZE)

        # Check neighboring cells for possible neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in grid_map:
                    for other_idx in grid_map[neighbor_cell]:
                        if other_idx == idx:
                            continue
                        # Compute pairwise distance
                        distance = np.linalg.norm(centers[idx] - centers[other_idx])
                        # Update closest distances for both circles
                        if distance < closest_distances[idx]:
                            closest_distances[idx] = distance
                        if distance < closest_distances[other_idx]:
                            closest_distances[other_idx] = distance

    # Limit radii to half of the closest distance between circles and
    # the distance to the edge of the square
    for idx in range(n):
        radii[idx] = min(radii[idx], closest_distances[idx] / 2.0)

    return radii
# EVOLVE-BLOCK-END