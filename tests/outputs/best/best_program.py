

# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using modular and vectorized logic"""
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

    # Place circles in a structured pattern
    center = [0.5, 0.5]  # Center of main circle

    # Assign initial positions
    centers[0] = center  # Central circle

    # Helper function to place a circle ring
    def place_ring(base_center, ring_radius, start_index, num_circles):
        for i in range(num_circles):
            angle = 2 * np.pi * i / num_circles
            x = base_center[0] + ring_radius * np.cos(angle)
            y = base_center[1] + ring_radius * np.sin(angle)
            centers[start_index + i] = [x, y]

    # First ring of 8 smaller circles (radius = 0.3)
    place_ring(center, 0.3, start_index=1, num_circles=8)

    # Second ring of 16 even smaller circles (radius = 0.7)
    place_ring(center, 0.7, start_index=9, num_circles=16)

    # Clip centers to ensure all circles remain within unit square (safety margin of 0.01 units)
    centers = np.clip(centers, 0.01, 0.99)

    # Compute radii that avoid collisions and maintain containment in the square
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
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

    # Vectorized calculation of max radius due to square boundary constraints
    min_to_sides = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1 - centers[:, 0], 1 - centers[:, 1])
    )
    radii = min_to_sides

    # Iteratively adjust radii to prevent circles from overlapping
    for i in range(n):
        for j in range(i + 1, n):
            # Distance between i and j centers
            dist = np.linalg.norm(centers[i] - centers[j])

            # If sum of radii would cause overlap, scale both down
            if radii[i] + radii[j] > dist:
                scale_factor = dist / (radii[i] + radii[j])
                radii[i] *= scale_factor
                radii[j] *= scale_factor

    return radii


# EVOLVE-BLOCK-END