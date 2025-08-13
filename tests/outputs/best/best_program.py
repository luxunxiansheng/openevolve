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

    # Place central circle
    centers[0] = [0.5, 0.5]

    # Place 8 circles around the central circle in a first ring
    ring_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring_positions = 0.5 + 0.3 * np.array([np.cos(ring_angles), np.sin(ring_angles)]).T
    centers[1:9] = ring_positions

    # Place 17 additional circles in an outer ring
    outer_ring_angles = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    outer_positions = 0.5 + 0.7 * np.array([np.cos(outer_ring_angles), np.sin(outer_ring_angles)]).T
    centers[9:] = outer_positions

    # Enforce square boundaries (keep a 1% margin)
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii based on circle packing rules
    radii = compute_max_radii(centers)

    # Calculate sum of radii
    sum_of_radii = np.sum(radii)

    return centers, radii, sum_of_radii


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
    # Start with maximum possible radius assuming no overlaps
    radii = np.ones(n)

    # Step 1: Limit each circle's radius due to square boundary constraints
    x, y = centers[:, 0], centers[:, 1]
    min_distance_to_boundary = np.minimum(np.minimum(np.minimum(x, y), 1 - x), 1 - y)
    radii = min_distance_to_boundary

    # Step 2: Limit each circle's radius due to inter-circle constraints
    # This nested loop enforces the condition: r_i + r_j <= distance between centers of circle i and j
    # We iterate over all pairs of circles (i, j) where i < j
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                # Reduce both radii in the same proportional way to ensure the sum of radii <= distance
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
# EVOLVE-BLOCK-END