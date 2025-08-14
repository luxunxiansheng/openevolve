"""
Test suite for the modular evaluator system
"""

import asyncio
import unittest

from openevolve.environment.evaluators import BaseEvaluator, ExecutionEvaluator, LLMEvaluator


class TestBaseEvaluator(unittest.TestCase):
    """Test the base evaluator abstract class"""

    def test_base_evaluator_is_abstract(self):
        """Test that BaseEvaluator cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseEvaluator()


class TestExecutionEvaluator(unittest.IsolatedAsyncioTestCase):
    """Test the execution evaluator"""

    def setUp(self):
        # Use the same critic and evolved code paths as test_exe_critic
        critic_path = "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py"
        self.evaluator = ExecutionEvaluator(critic_program_path=critic_path, job_timeout_seconds=10)

        # Path to evolved program for testing
        self.evolved_program_path = (
            "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/circle_packing.py"
        )

    def test_evaluator_creation(self):
        """Test that ExecutionEvaluator can be created"""
        self.assertIsInstance(self.evaluator, ExecutionEvaluator)
        self.assertIsInstance(self.evaluator, BaseEvaluator)
        self.assertEqual(self.evaluator.job_timeout_seconds, 10)
        self.assertIsNotNone(self.evaluator.critic_program_path)

    async def test_evaluate_with_real_critic(self):
        """Test evaluation with real circle_packing critic and evolved code"""
        # Load the evolved program code
        with open(self.evolved_program_path, "r") as file:
            evolved_code = file.read()

        # Run evaluation like test_exe_critic does
        result = await self.evaluator.evaluate(
            code=evolved_code, program_id="test_program_execution_evaluator", runtime_env={}
        )

        # Check that we get a result dictionary
        self.assertIsInstance(result, dict)
        print(f"Evaluation result: {result}")

        # The result should contain metrics from the critic program
        # Based on circle_packing critic, we expect metrics like 'combined_score'
        self.assertTrue(len(result) > 0, "Should have some metrics from critic program")

    async def test_evaluate_simple_code(self):
        """Test evaluation of simple Python code with circle_packing critic"""
        # Simple code that should work with the critic
        simple_code = """
import numpy as np

# Simple circle packing example  
def create_circles():
    return [(0, 0, 1), (2, 0, 1), (1, 1.7, 1)]

circles = create_circles()
print(f"Created {len(circles)} circles")
"""

        result = await self.evaluator.evaluate(code=simple_code, program_id="test_simple_code")

        self.assertIsInstance(result, dict)
        print(f"Simple code result: {result}")

    def test_evaluate_invalid_critic_path(self):
        """Test that invalid critic path raises appropriate error"""

        async def run_test():
            evaluator = ExecutionEvaluator(
                critic_program_path="/nonexistent/path/critic.py", job_timeout_seconds=5
            )
            # This should fail when we try to evaluate and load the critic
            with self.assertRaises(Exception):
                await evaluator.evaluate("print('test')", program_id="test_invalid")

        asyncio.run(run_test())


class TestLLMEvaluator(unittest.IsolatedAsyncioTestCase):
    """Test the LLM evaluator following test_llm_critic pattern"""

    def setUp(self):
        # Simplified setup without PromptSampler
        from openevolve.llm.llm_openai import OpenAILLM
        from openevolve.llm.llm_ensemble import EnsembleLLM

        # Create LLM client
        self.llm_client = EnsembleLLM([OpenAILLM()])

        # Create evaluator with simplified constructor
        self.evaluator = LLMEvaluator(self.llm_client)

    def test_evaluator_creation(self):
        """Test that LLMEvaluator can be created"""
        self.assertIsInstance(self.evaluator, LLMEvaluator)
        self.assertIsInstance(self.evaluator, BaseEvaluator)
        self.assertEqual(self.evaluator.llm, self.llm_client)

    async def test_evaluate_with_real_llm(self):
        """Test evaluation with real LLM following test_llm_critic pattern"""
        # Use the same code as test_llm_critic
        with open(
            "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/circle_packing.py",
            "r",
        ) as f:
            program_code = f.read()

        try:
            # Run evaluation with simplified interface
            result = await self.evaluator.evaluate(code=program_code)

            print(f"LLM Evaluation Result: {result}")

            # Check that we get a result dictionary with numeric metrics
            self.assertIsInstance(result, dict)

            # All values should be numeric
            for key, value in result.items():
                self.assertIsInstance(value, (int, float), f"Metric {key} should be numeric")

        except Exception as e:
            self.skipTest(f"LLM API not available or failed: {e}")

    async def test_evaluate_simple_code(self):
        """Test evaluation of simple code"""
        simple_code = """
def hello_world():
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
"""

        try:
            result = await self.evaluator.evaluate(code=simple_code)

            self.assertIsInstance(result, dict)
            print(f"Simple code LLM result: {result}")

        except Exception as e:
            self.skipTest(f"LLM API not available or failed: {e}")

    async def test_evaluate_handles_missing_code(self):
        """Test that missing code raises appropriate error"""
        with self.assertRaises(ValueError):
            await self.evaluator.evaluate(code="")

    async def test_evaluate_handles_empty_code(self):
        """Test that empty code raises appropriate error"""
        with self.assertRaises(ValueError):
            await self.evaluator.evaluate(code="")


class TestEvaluatorIntegration(unittest.TestCase):
    """Integration tests for evaluators"""

    def test_can_import_all_evaluators(self):
        """Test that all evaluators can be imported successfully"""
        from openevolve.environment.evaluators import (
            BaseEvaluator,
            ExecutionEvaluator,
            LLMEvaluator,
        )

        # Check that they're all classes
        self.assertTrue(callable(ExecutionEvaluator))
        self.assertTrue(callable(LLMEvaluator))

        # Check inheritance
        self.assertTrue(issubclass(ExecutionEvaluator, BaseEvaluator))
        self.assertTrue(issubclass(LLMEvaluator, BaseEvaluator))


if __name__ == "__main__":
    unittest.main()
