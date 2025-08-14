from openevolve.critic.critic import EvaluationResult, Critic


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

    # Target value from the paper


TARGET_VALUE = 2.635  # AlphaEvolve result for n=26


class CirclePackingCritic(Critic):
    """
    A critic for evaluating circle packing solutions.
    This critic evaluates the solution based on the sum of radii and checks for overlaps.
    """

    async def evaluate(self, **kwargs) -> EvaluationResult:

        try:

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
                mismatch_warning = f"Warning: Reported sum {reported_sum_radii} doesn't match calculated sum {sum_radii}"
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
                artifacts["sum_mismatch"] = (
                    f"Reported: {reported_sum_radii:.6f}, Calculated: {sum_radii:.6f}"
                )

            # Add successful packing stats for good solutions
            if valid and target_ratio > 0.95:  # Near-optimal solutions
                artifacts["stdout"] = (
                    f"Excellent packing! Achieved {target_ratio:.1%} of target value"
                )
                artifacts["radius_stats"] = (
                    f"Min: {validation_details['min_radius']:.6f}, Max: {validation_details['max_radius']:.6f}, Avg: {validation_details['avg_radius']:.6f}"
                )

            metrics = {
                "combined_score": combined_score,
            }

            # Importantly, log the artifacts and metrics otherwise the critic will not work
            # This is necessary for the critic to function correctly
            self.log_artifact(dict(artifacts))
            self.log_metrics(dict(metrics))

            return EvaluationResult(metrics=metrics, artifacts=artifacts)
        except Exception as e:
            # Handle any exceptions that occur during evaluation
            import traceback

            error_message = f"Error during evaluation: {str(e)}"
            full_traceback = traceback.format_exc()

            print(f"Error occurred: {error_message}")
            print(f"Full traceback:\n{full_traceback}")

            artifacts = {"error": error_message, "validity": 0.0, "traceback": full_traceback}
            metrics = {
                "combined_score": 0.0,
            }

            # Log the error artifacts and metrics
            self.log_artifact(dict(artifacts))
            self.log_metrics(dict(metrics))

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
