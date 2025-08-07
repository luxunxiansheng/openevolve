# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import logging
from cv2 import log
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
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
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

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


from openevolve.critic.critic import EvaluationResult, Critic

        # Target value from the paper
TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

class CirclePackingCritic(Critic):
    """
    A critic for evaluating circle packing solutions.
    This critic evaluates the solution based on the sum of radii and checks for overlaps.
    """


    async def evaluate(self, **kwargs) -> EvaluationResult:

        
        centers = run_packing()[0]  # Get centers from the packing function
        radii = run_packing()[1]  # Get radii from the packing function
        reported_sum_radii = run_packing()[2]  # Get sum of radii

        # Calculate the sum of radii
        sum_radii = sum(radii)

        # Check for overlaps
        valid, validation_details = self._validate_packing(centers, radii)

                # Make sure reported_sum matches the calculated sum
        sum_mismatch = abs(sum_radii - reported_sum_radii) > 1e-6
        if sum_mismatch:
            mismatch_warning = (
                f"Warning: Reported sum {reported_sum_radii} doesn't match calculated sum {sum_radii}"
            )
            print(mismatch_warning)

        # Target ratio (how close we are to the target)
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        artifacts = {
           
            "packing_summary": f"Sum of radii: {sum_radii:.6f}/{TARGET_VALUE} = {target_ratio:.4f}",
            "validation_report": f"Valid: {valid}, Violations: {len(validation_details.get('boundary_violations', []))} boundary, {len(validation_details.get('overlaps', []))} overlaps",
        }

            # Add sum mismatch warning if present
        if sum_mismatch:
            artifacts["sum_mismatch"] = f"Reported: {reported_sum_radii:.6f}, Calculated: {sum_radii:.6f}"

        # Add successful packing stats for good solutions
        if valid and target_ratio > 0.95:  # Near-optimal solutions
            artifacts["stdout"] = f"Excellent packing! Achieved {target_ratio:.1%} of target value"
            artifacts[
                "radius_stats"
            ] = f"Min: {validation_details['min_radius']:.6f}, Max: {validation_details['max_radius']:.6f}, Avg: {validation_details['avg_radius']:.6f}"


        metrics = {
            "sum_radii": reported_sum_radii,
            "target_ratio": target_ratio,
            "validity": validity,
            "combined_score": combined_score,
            }

        
        # Importantly, log the artifacts and metrics otherwise the critic will not work
        # This is necessary for the critic to function correctly
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

        # Check if circles are inside the unit square
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
                violation = (
                    f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
                )
                validation_details["boundary_violations"].append(violation)
                print(violation)

        # Check for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                    overlap = f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
                    validation_details["overlaps"].append(overlap)
                    print(overlap)

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
