# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles with improved readability and correctness"""

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

    # Central circle
    centers[0] = [0.5, 0.5]

    # Function to place a circular ring of circles around a center point
    def place_circular_ring(start_index, center_position, ring_radius, num_circles):
        """
        Places a circular ring of `num_circles` circles at a given radius around
        the specified `center_position`, starting from index `start_index`.

        Args:
            start_index: Index in the centers array to begin placing the ring
            center_position: Tuple (x, y) of the ring's center
            ring_radius: Radius at which to place the circle ring
            num_circles: Number of circles in the ring
        """
        center_x, center_y = center_position
        for circle_idx in range(num_circles):
            angle = 2 * np.pi * circle_idx / num_circles
            x = center_x + ring_radius * np.cos(angle)
            y = center_y + ring_radius * np.sin(angle)
            centers[start_index + circle_idx] = [x, y]

    # Place first ring of 8 circles around center
    place_circular_ring(start_index=1, center_position=(0.5, 0.5), ring_radius=0.3, num_circles=8)

    # Place second ring of 17 circles around center to complete the full count
    place_circular_ring(start_index=9, center_position=(0.5, 0.5), ring_radius=0.7, num_circles=17)

    # Clip all circle centers to ensure they fit within the unit square (with 1% margin)
    centers = np.clip(centers, 0.01, 0.99)

    # Validate that all circle centers have been properly initialized
    if np.all(centers == 0.0):
        raise ValueError("Error: Some circle centers remain unset in the layout.")
    
    # Compute maximum valid radii to ensure no overlaps and fit in unit square
    radii = compute_max_radii(centers)

    # Sum the computed radii
    sum_of_radii = np.sum(radii)

    return centers, radii, sum_of_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they do not overlap and remain within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Ensure circles do not overflow beyond square boundaries by limiting them to minimum edge distance
    for circle_idx in range(n):
        x, y = centers[circle_idx]
        boundary = np.array([x, y, 1.0 - x, 1.0 - y])  # [left, bottom, right, top]
        radii[circle_idx] = np.min(boundary)

    # Resolve overlaps: Adjust radii between every pair until no overlaps exist
    for circle_a in range(n):
        for circle_b in range(circle_a + 1, n):
            dx, dy = centers[circle_a] - centers[circle_b]
            distance = np.sqrt(dx**2 + dy**2)

            if radii[circle_a] + radii[circle_b] > distance:
                scale_factor = distance / (radii[circle_a] + radii[circle_b])
                radii[circle_a] *= scale_factor
                radii[circle_b] *= scale_factor

    return radii


# EVOLVE-BLOCK-END