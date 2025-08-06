"""
Evaluator for circle packing example (n=26) with improved timeout handling
Enhanced with artifacts to demonstrate execution feedback
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle




class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
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
                overlap = (
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
                )
                validation_details["overlaps"].append(overlap)
                print(overlap)

    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
    )
    validation_details["is_valid"] = is_valid

    return is_valid, validation_details




def evaluate(program_path):
   