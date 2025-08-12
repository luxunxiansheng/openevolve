# EVOLVE-BLOCK-START
"""
Optimized and well-documented circle-packing algorithm for 26 circles in a unit square  
Maximizing the sum of radii with improved clarity, maintainability, and correctness.  
"""

import numpy as np
from collections import defaultdict

# Configuration for circle placement - make configurable for maintainability
RING_CONFIG = [
    (0.3, 8),   # Ring radius and number of circles in first ring
    (0.7, 17),  # Ring radius and number of circles in second ring
]

# Grid partitioning parameter
GRID_CELL_SIZE = 0.2  # Partition square into 5x5 grid


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

    # Place central circle
    centers[0] = [0.5, 0.5]

    def _place_ring(start_index, center, ring_radius, num_circles):
        """
        Place a uniform ring of circles around a given center.

        Args:
            start_index: First index to assign positions in centers array
            center: (x, y) position of the ring center
            ring_radius: Radius of the ring (distance from center to circle)
            num_circles: Number of circles in the ring
        """
        cx, cy = center
        for idx in range(num_circles):
            angle = 2 * np.pi * idx / num_circles
            x = cx + ring_radius * np.cos(angle)
            y = cy + ring_radius * np.sin(angle)
            centers[start_index + idx] = [x, y]

    # Place rings using pre-defined configuration
    current_start_index = 1
    for i, (radius, num_circles) in enumerate(RING_CONFIG):
        _place_ring(
            start_index=current_start_index,
            center=(0.5, 0.5),
            ring_radius=radius,
            num_circles=num_circles
        )
        current_start_index += num_circles

    # Ensure circles stay within margin
    centers = np.clip(centers, 0.01, 0.99)

    # Compute optimal radii
    radii = compute_optimal_radii(centers)

    return centers, radii, np.sum(radii)


def compute_optimal_radii(centers):
    """
    Calculate optimal radii to satisfy no overlaps and boundary constraints.

    Args:
        centers: np.array of (n, 2) circle positions

    Returns:
        np.array of valid radii for all circles
    """
    n = centers.shape[0]

    # Initialize with minimum edge distance as preliminary radii
    edge_min = np.minimum(centers[:, 0], centers[:, 1])
    edge_max = np.minimum(1 - centers[:, 0], 1 - centers[:, 1])
    radii = np.minimum(edge_min, edge_max)

    # Use grid partitioning for efficient spatial comparisons
    grid_map = defaultdict(list)

    for idx, (x, y) in enumerate(centers):
        cell_x = int(x / GRID_CELL_SIZE)
        cell_y = int(y / GRID_CELL_SIZE)
        grid_map[(cell_x, cell_y)].append(idx)

    # Find minimum circle-to-circle distance for each circle
    closest_distances = [np.inf] * n

    for idx, (x, y) in enumerate(centers):
        cell_x = int(x / GRID_CELL_SIZE)
        cell_y = int(y / GRID_CELL_SIZE)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in grid_map:
                    for other_idx in grid_map[neighbor_cell]:
                        if other_idx == idx:
                            continue
                        distance = np.linalg.norm(centers[idx] - centers[other_idx])
                        if distance < closest_distances[idx]:
                            closest_distances[idx] = distance
                        if distance < closest_distances[other_idx]:
                            closest_distances[other_idx] = distance

    # Limit radii to half of closest distance to prevent overlap
    for idx in range(n):
        radii[idx] = min(radii[idx], closest_distances[idx] / 2.0)

    return radii
# EVOLVE-BLOCK-END