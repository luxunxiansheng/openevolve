

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

    # Place 8 circles in a ring around the center
    place_ring(centers, 1, 0.3, 8)

    # Place 16 circles in a larger outer ring
    place_ring(centers, 9, 0.7, 16)

    # Ensure all centers are within the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute the maximum valid radii
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def place_ring(centers, start_index, radius, num_circles):
    """
    Place a ring of circles around a central point.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        start_index: Index to start placing circles
        radius: Distance from the center of the ring
        num_circles: Number of circles in the ring
    """
    for i in range(num_circles):
        angle = 2 * np.pi * i / num_circles
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        centers[start_index + i] = [x, y]


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

    # Compute initial radii based on distance to square borders
    x = centers[:, 0]
    y = centers[:, 1]
    border_distances = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    radii = border_distances.copy()

    # Adjust radii to avoid overlap between circles
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate distance between centers
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END