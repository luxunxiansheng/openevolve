"""
Integration test for ExecutionEvaluator with real Ray cluster and critic program

Tests the execution evaluator against a real Ray cluster without using mocks or dummies.
Requires a Ray cluster to be running on localhost:8265.
"""

import asyncio
import tempfile
import unittest
import os

try:
    from ray.job_submission import JobSubmissionClient, JobStatus

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from opencontext.environment.program_evaluation.execution_evaluator import ExecutionEvaluator


class TestExecutionEvaluatorIntegration(unittest.TestCase):
    """
    Integration test for ExecutionEvaluator using real Ray cluster

    This test creates a critic program following the real pattern from circle_packing example
    and evaluates code with proper working directory setup for imports.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment with a critic program following real patterns"""
        # Create a temporary critic program for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.critic_path = os.path.join(cls.temp_dir, "test_critic.py")

        # Critic program following the pattern from circle_packing_with_artifacts_new/critic.py
        critic_content = '''
"""
Test critic program following the real pattern from circle_packing example
"""

# Import the base evaluator - this works because we set working_dir properly
from opencontext.environment.program_evaluation.base_evaluator import BaseEvaluator, EvaluationResult
import logging

logger = logging.getLogger(__name__)


class MathFunctionCritic(BaseEvaluator):
    """
    A critic for evaluating basic math functions like add and multiply.
    This follows the same pattern as CirclePackingCritic.
    """

    async def evaluate(
        self, code: str = "", language: str = "python", **kwargs
    ) -> EvaluationResult:
        """
        Evaluate the math functions that should be defined in the global scope
        """
        try:
            # Test if the add function exists and works correctly
            if 'add' in globals():
                result1 = add(2, 3)
                correctness = 1.0 if result1 == 5 else 0.0
            else:
                correctness = 0.0
                print("ERROR: add function not found")
            
            # Test if the multiply function exists and works correctly  
            if 'multiply' in globals():
                result2 = multiply(4, 5)
                performance = 1.0 if result2 == 20 else 0.0
            else:
                performance = 0.0
                print("ERROR: multiply function not found")
            
            # Simple readability check (functions exist)
            readability = 1.0 if 'add' in globals() and 'multiply' in globals() else 0.0
            
            # Maintainability based on code quality (simple heuristic)
            maintainability = 0.8  # Default decent score
            
            # Create metrics dict
            metrics = {
                "correctness": correctness,
                "add_result": result1 if 'add' in globals() else None,
                "multiply_result": result2 if 'multiply' in globals() else None,
                "performance": performance, 
                "readability": readability,
                "maintainability": maintainability
            }
            
            # Create artifacts with evaluation details
            artifacts = {
                "evaluation_summary": f"Functions tested: add({'found' if 'add' in globals() else 'missing'}), multiply({'found' if 'multiply' in globals() else 'missing'})",
                "test_results": f"add(2,3)={'pass' if correctness > 0.5 else 'fail'}, multiply(4,5)={'pass' if performance > 0.5 else 'fail'}",
                "notes": "Basic math function evaluation completed"
            }
            
            # Output metrics in the format expected by ExecutionEvaluator
            for key, value in metrics.items():
                print(f"{key}: {value}")
                
            # Output artifacts in the format expected by ExecutionEvaluator  
            for key, value in artifacts.items():
                print(f"Artifact {key}: {value}")
            
            return EvaluationResult(metrics=metrics, artifacts=artifacts)
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            print(f"ERROR in evaluation: {e}")
            
            # Output zero metrics on error
            error_metrics = {
                "correctness": 0.0,
                "performance": 0.0,
                "readability": 0.0,
                "maintainability": 0.0
            }
            
            # Output error artifacts
            error_artifacts = {
                "error_message": str(e),
                "evaluation_status": "failed"
            }
            
            for key, value in error_metrics.items():
                print(f"{key}: {value}")
                
            for key, value in error_artifacts.items():
                print(f"Artifact {key}: {value}")
            
            return EvaluationResult(metrics=error_metrics, artifacts=error_artifacts)


# This is the main execution part that runs when the script is executed by Ray
if __name__ == "__main__":
    import asyncio
    
    # Create and run the critic just like in the circle packing example
    critic = MathFunctionCritic()
    result = asyncio.run(critic.evaluate())
    
    print(f"Final evaluation completed with metrics: {result.metrics}")
'''

        with open(cls.critic_path, "w") as f:
            f.write(critic_content)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up individual test"""
        self.evaluator = ExecutionEvaluator(
            critic_program_path=self.critic_path,
            ray_head_ip="http://127.0.0.1:8265",
            job_timeout_seconds=60,
        )

    def _get_runtime_env(self):
        """Get runtime environment with proper working directory for imports"""
        return {
            "working_dir": "/workspaces/opencontext",  # Set to repo root so imports work
            "pip": [],  # Don't install additional packages since we have everything
        }

    @unittest.skipUnless(RAY_AVAILABLE, "Ray not available - install ray package")
    def test_evaluate_simple_math_functions_with_working_dir(self):
        """Test evaluation of simple math functions with proper working directory"""
        # Simple code that should pass the critic's tests
        test_code = '''
def add(a, b):
    """Add two numbers"""
    return a + b

def multiply(a, b):
    """Multiply two numbers"""
    return a * b
'''

        # Run the evaluation with proper runtime environment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.evaluator.evaluate(
                    code=test_code, language="python", runtime_env=self._get_runtime_env()
                )
            )

            # Verify we got a result
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.metrics)

            # Check that we have expected metrics
            expected_metrics = ["correctness", "performance", "readability", "maintainability"]
            for metric in expected_metrics:
                self.assertIn(metric, result.metrics, f"Missing metric: {metric}")
                self.assertIsInstance(
                    result.metrics[metric], (int, float), f"Metric {metric} should be numeric"
                )
                self.assertGreaterEqual(
                    result.metrics[metric], 0.0, f"Metric {metric} should be >= 0"
                )
                self.assertLessEqual(result.metrics[metric], 1.0, f"Metric {metric} should be <= 1")

            # For correct functions, we should get high scores
            self.assertGreaterEqual(
                result.metrics["correctness"],
                0.8,
                "Correctness should be high for correct functions",
            )
            self.assertGreaterEqual(
                result.metrics["performance"],
                0.8,
                "Performance should be high for correct functions",
            )

            # Verify artifacts were extracted
            self.assertIsNotNone(result.artifacts, "Should have artifacts")
            self.assertIsInstance(result.artifacts, dict, "Artifacts should be a dictionary")

            # Check for expected artifacts from our critic
            expected_artifacts = ["evaluation_summary", "test_results", "notes"]
            for artifact in expected_artifacts:
                self.assertIn(artifact, result.artifacts, f"Missing artifact: {artifact}")
                self.assertIsInstance(
                    result.artifacts[artifact], str, f"Artifact {artifact} should be string"
                )

            print(f"✅ Evaluation successful: {result.metrics}")
            print(f"✅ Artifacts extracted: {list(result.artifacts.keys())}")
            if result.artifacts:
                sample_artifacts = []
                for k, v in list(result.artifacts.items())[:2]:
                    v_str = str(v)
                    display_value = v_str[:50] + "..." if len(v_str) > 50 else v_str
                    sample_artifacts.append((k, display_value))
                print(f"✅ Sample artifacts: {sample_artifacts}")

        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()

    @unittest.skipUnless(RAY_AVAILABLE, "Ray not available - install ray package")
    def test_evaluate_broken_code_with_working_dir(self):
        """Test evaluation of code that should fail the critic's tests"""
        # Code with incorrect implementations
        broken_code = '''
def add(a, b):
    """This add function is broken"""
    return a - b  # Wrong operation!

def multiply(a, b):
    """This multiply function is broken"""  
    return a + b  # Wrong operation!
'''

        # Run the evaluation with proper runtime environment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.evaluator.evaluate(
                    code=broken_code, language="python", runtime_env=self._get_runtime_env()
                )
            )

            # Verify we got a result
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.metrics)

            # For broken functions, correctness and performance should be low
            self.assertLessEqual(
                result.metrics["correctness"], 0.2, "Correctness should be low for broken functions"
            )
            self.assertLessEqual(
                result.metrics["performance"], 0.2, "Performance should be low for broken functions"
            )

            print(f"✅ Broken code evaluation as expected: {result.metrics}")

        finally:
            loop.close()
