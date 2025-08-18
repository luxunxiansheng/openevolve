"""
Test for ProgramEvolutionEnv main methods

Tests the main methods of ProgramEvolutionEnv: reset, step, render, close
Uses real LLM and ExecutionEvaluator without mocks or dummies.
"""

import tempfile
import unittest
import os
import logging
import sys

try:
    from ray.job_submission import JobSubmissionClient, JobStatus

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from opencontext.common.actions import EvolutionAction, EvolutionMode
from opencontext.common.program import Program
from opencontext.environment.program_evolution_env import ProgramEvolutionEnv
from opencontext.environment import setup_environment_logging
from opencontext.llm.llm_openai import OpenAILLM
from opencontext.environment.program_evaluation.execution_evaluator import ExecutionEvaluator

# Set up logging for tests - configure to show all logger output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

# Also setup environment logging
setup_environment_logging(level="INFO", include_extra=True)

# Create a specific logger for tests and ensure it's visible
test_logger = logging.getLogger("test")
test_logger.setLevel(logging.INFO)
if not test_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("TEST: %(message)s"))
    test_logger.addHandler(handler)
    test_logger.propagate = False


class TestProgramEvolutionEnvMainMethods(unittest.TestCase):
    """
    Test for ProgramEvolutionEnv main methods using real services
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment with a simple critic program"""
        # Create a temporary critic program for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.critic_path = os.path.join(cls.temp_dir, "simple_critic.py")

        # Simple critic program that outputs basic metrics
        critic_content = """
print("correctness: 0.8")
print("performance: 0.7")
print("readability: 0.9")
print("maintainability: 0.8")
"""
        with open(cls.critic_path, "w") as f:
            f.write(critic_content)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up individual test"""
        self.logger = test_logger

        self.llm = OpenAILLM()
        self.execution_evaluator = ExecutionEvaluator(
            critic_program_path=self.critic_path,
            ray_head_ip="http://127.0.0.1:8265",
            job_timeout_seconds=60,
            logger=self.logger,
        )

    def test_reset(self):
        """Test reset method"""
        self.logger.info("Testing reset method")

        env = ProgramEvolutionEnv(self.llm, self.execution_evaluator, logger=self.logger)

        # Test reset
        obs, info = env.reset()

        # Verify observation structure
        self.assertIsInstance(obs, dict)
        self.assertIn("generated_program", obs)
        self.assertIn("evaluation_metrics", obs)
        self.assertIn("success", obs)

        # Verify initial values
        self.assertEqual(obs["generated_program"], "")
        self.assertEqual(obs["evaluation_metrics"], {})
        self.assertEqual(obs["success"], 0)

        # Verify info
        self.assertIsInstance(info, dict)
        self.assertIn("episode", info)
        self.assertEqual(info["episode"], 1)

        # Test multiple resets increment episode count
        obs2, info2 = env.reset()
        self.assertEqual(info2["episode"], 2)

        self.logger.info("Reset method test completed successfully")

    @unittest.skipUnless(RAY_AVAILABLE, "Ray not available")
    def test_step(self):
        """Test step method with valid action"""
        self.logger.info("Testing step method")

        env = ProgramEvolutionEnv(self.llm, self.execution_evaluator, logger=self.logger)
        env.reset()

        # Create a simple program to evolve
        current_program = Program(
            id="test-1", code="def hello():\n    print('Hello')", language="python"
        )

        # Create evolution action
        action = EvolutionAction(
            goal="Add a docstring",
            instructions=["Add a brief docstring"],
            current_program=current_program,
            mode=EvolutionMode.FULL_REWRITE,
        )

        # Execute step
        obs, reward, done, truncated, info = env.step(action)
        self.logger.info(f"Step result - Success: {obs.get('success')}, Reward: {reward}")

        # Verify response structure
        self.assertIsInstance(obs, dict)
        self.assertIn("generated_program", obs)
        self.assertIn("evaluation_metrics", obs)
        self.assertIn("success", obs)

        # Should be successful
        self.assertEqual(obs["success"], 1)

        # Should have generated program
        self.assertIsInstance(obs["generated_program"], str)
        self.assertGreater(len(obs["generated_program"]), 0)

        # Should have evaluation metrics
        self.assertIsInstance(obs["evaluation_metrics"], dict)

        # Reward should be a float
        self.assertIsInstance(reward, float)

        self.logger.info("Step method test completed successfully")

    def test_step_invalid_action(self):
        """Test step method with invalid action type"""
        self.logger.info("Testing step method with invalid action")

        env = ProgramEvolutionEnv(self.llm, self.execution_evaluator, logger=self.logger)
        env.reset()

        # Try to step with invalid action (not EvolutionAction)
        # The environment should now handle this gracefully and return an error response
        obs, reward, done, truncated, info = env.step("invalid")  # type: ignore

        # Should return error response
        self.assertEqual(obs["success"], 0)
        self.assertEqual(reward, 0.0)
        self.assertIn("error", info)
        self.assertIn("Invalid action type", info["error"])
        self.logger.info("Invalid action handled gracefully")

        self.logger.info("Invalid action test completed successfully")

    def test_render(self):
        """Test render method"""
        env = ProgramEvolutionEnv(self.llm, self.execution_evaluator, logger=self.logger)

        # Render should not raise exception
        try:
            env.render()
        except Exception as e:
            self.fail(f"render() raised {e} unexpectedly")

    def test_close(self):
        """Test close method"""
        env = ProgramEvolutionEnv(self.llm, self.execution_evaluator, logger=self.logger)

        # Close should not raise exception
        try:
            env.close()
        except Exception as e:
            self.fail(f"close() raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
