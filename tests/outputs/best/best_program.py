# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles with improved readability and maintainability"""
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

    # Place 8 circles in an inner ring
    place_ring(centers, start_index=1, num_circles=8, ring_radius=0.3)

    # Place 16 circles in an outer ring
    place_ring(centers, start_index=9, num_circles=16, ring_radius=0.7)

    # Ensure centers stay within the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii based on boundary and overlap constraints
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def place_ring(centers, start_index, num_circles, ring_radius):
    """
    Places a ring of circles around a central point.

    Args:
        centers: Array to store circle centers.
        start_index: Starting index in the centers array.
        num_circles: Number of circles in the ring.
        ring_radius: Distance from the central point to the ring.
    """
    for i in range(num_circles):
        angle = 2 * np.pi * i / num_circles
        x = 0.5 + ring_radius * np.cos(angle)
        y = 0.5 + ring_radius * np.sin(angle)
        centers[start_index + i] = [x, y]


def compute_pairwise_distances(centers):
    """
    Compute pairwise distances between all circle centers using NumPy broadcasting.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n, n) with pairwise distances
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def compute_max_radii(centers):
    """
    Computes the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Step 1: Limit radii by distance to square boundaries
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Step 2: Precompute all pairwise distances
    distances = compute_pairwise_distances(centers)

    # Step 3: Limit radii by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = distances[i, j]
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    return construct_packing()


from openevolve.critic.critic import EvaluationResult, Critic


class CirclePackingCritic(Critic):
    """
    A critic for evaluating circle packing solutions.
    This critic evaluates the solution based on the sum of radii and checks for overlaps.
    """

    async def evaluate(self, **kwargs) -> EvaluationResult:
        artifacts = {}
        metrics = {}

        # Retrieve all values in a single call
        centers, radii, sum_radii = run_packing()

        # Check for overlaps
        is_valid, validation_details = self._validate_packing(centers, radii)

        metrics = {
            "sum_radii": sum_radii,
            "overlaps": len(validation_details["overlaps"]) > 0,
        }

        artifacts = {"centers": centers, "radii": radii}

        # Log artifacts and metrics
        self.log_artifact(artifacts)
        self.log_metrics(metrics)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def _validate_packing(self, centers, radii):
        """
        Validate that circles don't overlap and are inside the unit square

        Args:
            centers: np.array of shape (n, 2) with (x, y) coordinates
            radii: np.array of shape (n) with radius of each circle

        Returns:
            Tuple of (is_valid: bool, validation_details: dict)
        """
        n = centers.shape[0]
        validation_details = {
            "total_circles": n,
            "boundary_violations": [],
            "overlaps": [],
            "min_radius": float(np.min(radii)),
            "max_radius": float(np.max(radii)),
            "avg_radius": float(np.mean(radii)),
        }

        # Step 1: Check if circles are inside the unit square
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
                violation = (
                    f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
                )
                validation_details["boundary_violations"].append(violation)

        # Step 2: Compute pairwise distances using helper function
        distances = compute_pairwise_distances(centers)

        # Step 3: Check for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = distances[i, j]
                if dist < radii[i] + radii[j] - 1e-6:
                    overlap = f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
                    validation_details["overlaps"].append(overlap)

        is_valid = (
            len(validation_details["boundary_violations"]) == 0
            and len(validation_details["overlaps"]) == 0
        )
        validation_details["is_valid"] = is_valid

        return is_valid, validation_details


if __name__ == "__main__":
    import asyncio

    critic = CirclePackingCritic()
    result = asyncio.run(critic.evaluate())
    print("Evaluation Result:", result.metrics)
    print("Artifacts:", result.artifacts)