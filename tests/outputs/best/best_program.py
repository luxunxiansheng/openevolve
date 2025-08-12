

# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles with enhanced performance"""

import numpy as np

def place_ring(centers, base_center, radius, start_idx, num_circles):
    """
    Places a ring of circles around the `base_center` with a given `radius`.

    Args:
        centers: numpy array to store circle centers.
        base_center: [x, y] center of the ring.
        radius: radial distance from `base_center`.
        start_idx: starting index to populate the centers array.
        num_circles: number of circles in the ring.
    """
    for i in range(num_circles):
        angle = 2 * np.pi * i / num_circles
        x = base_center[0] + radius * np.cos(angle)
        y = base_center[1] + radius * np.sin(angle)
        centers[start_idx + i] = [x, y]
    return centers


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
    center_pos = np.array([0.5, 0.5])  # Central position for structured pattern

    # Place the center circle
    centers[0] = center_pos

    # Place 8 circles in a closer ring around the center circle
    place_ring(centers, center_pos, 0.3, 1, 8)

    # Place 16 circles in a more distant ring
    place_ring(centers, center_pos, 0.7, 9, 16)

    # Ensure all circles are within the unit square's interior
    centers = np.clip(centers, 0.01, 0.99)

    # Compute the maximum allowable radii given position constraints
    radii = compute_max_radii(centers)

    # Sum of all computed radii
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
    # Step 1: Compute radii based on boundaries
    radii = compute_boundary_constraints(centers)

    # Step 2: Handle overlap constraints using optimized vectorization
    radii = handle_overlap_constraints(centers, radii)

    return radii


def compute_boundary_constraints(centers):
    """
    Calculate the maximum radius each circle can have based solely on the square's boundaries.
    This is the minimum distance to any of the square's edges.

    Args:
        centers: Array of circle centers of shape (n, 2)

    Returns:
        Array of boundary-limited radii of shape (n,)
    """
    return np.minimum(np.minimum(centers[:, 0], centers[:, 1]), np.minimum(1 - centers[:, 0], 1 - centers[:, 1]))


def handle_overlap_constraints(centers, radii):
    """
    Adjusts the radii to prevent overlaps between circles using a distance matrix approach.
    Each circle's radius is limited to half the minimal distance to any other circle,
    using fully vectorized operations for efficiency.

    Args:
        centers: Numpy array of circle centers (n x 2)
        radii: Numpy array of current radii for each circle (n-length)

    Returns:
        Numpy array of adjusted radii (n-length)
    """
    n = centers.shape[0]
    # Compute pairwise distances between all circle centers
    dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

    # Create a copy and set the diagonals to infinity to exclude self-distance
    dist_without_diagonal = dist_matrix.copy()
    np.fill_diagonal(dist_without_diagonal, np.inf)

    # Compute minimum distance to other circles for each circle
    min_distances = np.min(dist_without_diagonal, axis=1)

    # Adjust radii to ensure r_i + r_j <= d_ij for all pairs (i,j)
    radii = np.minimum(radii, min_distances / 2)

    return radii


# EVOLVE-BLOCK-END