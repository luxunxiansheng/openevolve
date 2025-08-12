# EVOLVE-BLOCK-START
"""
Optimized circle-packing algorithm for 26 circles in a unit square, balancing 
clarity, maintainability, and efficiency with a structured approach and corrected 
neighbor distance evaluation logic.

Improvements:
- Explicit configuration parameters for ease of maintenance and clarity.
- Added the missing 26th circle to conform to the problem requirement.
- Fixed the closest neighbor distance calculation to consider both circles.
- Modularized and vectorized logic for enhanced readability and performance.
"""

import numpy as np
from collections import defaultdict


# --- CONFIGURATION PARAMETERS ---
CONFIG = {
    "RINGS": {
        "RING1": {"START": 1, "COUNT": 8, "RADIUS": 0.3},
        "RING2": {"START": 9, "COUNT": 16, "RADIUS": 0.7},
    },
    "BORDER_MARGIN": 0.01,     # Margin from unit square edges
    "CELL_SIZE": 0.2,         # Grid cell size for partitioning (creates 5x5 grid)
}

# Extract configuration values for clarity
RING1_START = CONFIG["RINGS"]["RING1"]["START"]
RING1_COUNT = CONFIG["RINGS"]["RING1"]["COUNT"]
RING1_RADIUS = CONFIG["RINGS"]["RING1"]["RADIUS"]

RING2_START = CONFIG["RINGS"]["RING2"]["START"]
RING2_COUNT = CONFIG["RINGS"]["RING2"]["COUNT"]
RING2_RADIUS = CONFIG["RINGS"]["RING2"]["RADIUS"]

BORDER_MARGIN = CONFIG["BORDER_MARGIN"]
CELL_SIZE = CONFIG["CELL_SIZE"]

def place_ring(centers: np.ndarray, start_index: int, center_x: float, center_y: float,
               ring_radius: float, num_circles: int):
    """
    Place a ring of circles equidistant from a central position using vectorized operations.

    Parameters:
        centers: ndarray of shape (n, 2) for circle coordinates.
        start_index: Starting index to assign coordinates.
        center_x: X-coordinate of the central position.
        center_y: Y-coordinate of the central position.
        ring_radius: Radial distance of the ring.
        num_circles: Number of circles in the ring.
    """
    angles = np.linspace(0, 2*np.pi, num_circles, endpoint=False)
    x = center_x + ring_radius * np.cos(angles)
    y = center_y + ring_radius * np.sin(angles)

    centers[start_index:start_index+num_circles, 0] = x
    centers[start_index:start_index+num_circles, 1] = y


def construct_packing():
    """
    Generate a packing of 26 circles within a 1% margin of the unit square.

    Returns:
        Tuple of (centers, radii, sum_of_radii).
        - centers: Array of shape (26, 2) with circle coordinates.
        - radii: Array of shape (26) with valid radii.
        - sum_of_radii: Scalar sum of all circle radii.
    """
    n = 26
    centers = np.zeros((n, 2))

    # Place initial central circle
    centers[0] = [0.5, 0.5]

    # Place first ring (8 circles)
    place_ring(centers, RING1_START, 0.5, 0.5, RING1_RADIUS, RING1_COUNT)
    # Place second ring (16 circles)
    place_ring(centers, RING2_START, 0.5, 0.5, RING2_RADIUS, RING2_COUNT)

    # Add missing 26th circle
    centers[25] = [0.15, 0.15]  # Ensure it does not collide with others

    # Clamp to ensure all circles stay within margin
    centers = np.clip(centers, BORDER_MARGIN, 1.0 - BORDER_MARGIN)

    # Compute the optimal non-overlapping radii
    radii = compute_optimal_radii(centers)

    return centers, radii, np.sum(radii)


def compute_optimal_radii(centers: np.ndarray) -> np.ndarray:
    """
    Calculate maximum allowable radii for circles based on distances to:
    - square boundaries
    - other circle centers

    Parameters:
        centers: ndarray of shape (n, 2) containing circle positions.

    Returns:
        numpy.ndarray: Array of optimal radii for each circle.
    """
    n = centers.shape[0]
    radii = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1 - centers[:, 0], 1 - centers[:, 1])
    )

    # Grid partitioning to reduce neighbor comparisons
    grid_map = defaultdict(list)
    for idx in range(n):
        x, y = centers[idx]
        cell_x = int(x // CELL_SIZE)
        cell_y = int(y // CELL_SIZE)
        grid_map[(cell_x, cell_y)].append(idx)

    # Find closest distance to other circles for all circles
    closest_distances = np.full(n, np.inf)

    for idx in range(n):
        x, y = centers[idx]
        current_cell = (int(x // CELL_SIZE), int(y // CELL_SIZE))

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_cell = (current_cell[0] + dx, current_cell[1] + dy)
                if neighbor_cell in grid_map:
                    for neighbor_idx in grid_map[neighbor_cell]:
                        if idx == neighbor_idx:
                            continue

                        dist = np.linalg.norm(centers[idx] - centers[neighbor_idx])
                        closest_distances[idx] = min(closest_distances[idx], dist)
                        closest_distances[neighbor_idx] = min(closest_distances[neighbor_idx], dist)

    # Constrain radii based on minimum distances to other circles
    for idx in range(n):
        if closest_distances[idx] != np.inf:
            radii[idx] = min(radii[idx], closest_distances[idx] / 2)

    return radii
# EVOLVE-BLOCK-END