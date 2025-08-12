# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
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
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit radii to the minimum distance to any square boundary
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Limit radii to prevent overlap between any pairs of circles
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Euclidean distance between centers
            distance = np.linalg.norm(centers[i] - centers[j])

            # Scale both radii proportionally if the sum exceeds distance
            if radii[i] + radii[j] > distance:
                scale_factor = distance / (radii[i] + radii[j])
                radii[i] *= scale_factor
                radii[j] *= scale_factor

    return radii


# EVOLVE-BLOCK-END