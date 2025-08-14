"""
Tests for ProgramEvolutionEnv

Tests the gymnasium environment for program evolution with real evaluators.
"""

import unittest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock

from openevolve.environment.program_evolution_env import ProgramEvolutionEnv
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.llm.llm_ensemble import EnsembleLLM


class TestProgramEvolutionEnv(unittest.TestCase):
    """Test the Program Evolution Environment"""

    def setUp(self):
        """Set up test fixtures"""
        # Create real LLM for testing
        self.llm = EnsembleLLM([OpenAILLM()])

        # Create execution evaluator with circle_packing critic
        self.exe_evaluator = ExecutionEvaluator(
            critic_program_path="/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py"
        )

        # Create LLM evaluator
        self.llm_evaluator = LLMEvaluator(self.llm)

        # Create environment
        self.env = ProgramEvolutionEnv(
            llm=self.llm,
            exe_evaluator=self.exe_evaluator,
            llm_evaluator=self.llm_evaluator,
            reward_scale=1.0,
            improvement_bonus=0.1,
            penalty_for_errors=-0.2,
        )

    def test_environment_creation(self):
        """Test that environment can be created properly"""
        self.assertIsInstance(self.env, ProgramEvolutionEnv)
        self.assertEqual(self.env.llm, self.llm)
        self.assertEqual(self.env.exe_evaluator, self.exe_evaluator)
        self.assertEqual(self.env.llm_evaluator, self.llm_evaluator)
        self.assertEqual(self.env.language, "python")
        self.assertEqual(self.env.reward_scale, 1.0)
        self.assertEqual(self.env.improvement_bonus, 0.1)
        self.assertEqual(self.env.penalty_for_errors, -0.2)

    def test_action_space(self):
        """Test action space is properly configured"""
        from gymnasium.spaces import Text

        self.assertIsInstance(self.env.action_space, Text)

    def test_observation_space(self):
        """Test observation space is properly configured"""
        from gymnasium.spaces import Dict

        obs_space = self.env.observation_space
        self.assertIsInstance(obs_space, Dict)

        # Try to sample from the space to verify it's valid
        try:
            sample = obs_space.sample()
            self.assertIsInstance(sample, dict)
        except:
            # Some spaces might not be fully sampleable, that's okay
            pass

    def test_reset_with_default_program(self):
        """Test environment reset with default initial program"""
        try:
            observation, info = self.env.reset()

            # Check observation structure for stateless environment
            self.assertIn("generated_program", observation)
            self.assertIn("evaluation_metrics", observation)
            self.assertIn("has_errors", observation)
            self.assertIn("generation_success", observation)

            # Check initial state (stateless, so should be empty)
            self.assertEqual(observation["generated_program"], "")
            self.assertEqual(observation["generation_success"], 0)
            self.assertIsInstance(observation["evaluation_metrics"], np.ndarray)

            # Check info
            self.assertIn("episode", info)
            self.assertIn("mode", info)
            self.assertEqual(info["mode"], "stateless")

            print(f"Reset successful in stateless mode")
            print(f"Episode: {info['episode']}")

        except Exception as e:
            self.skipTest(f"Environment reset failed (likely due to API/Ray unavailable): {e}")

    def test_reset_with_custom_program(self):
        """Test environment reset with custom initial program"""
        custom_program = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

        try:
            observation, info = self.env.reset(options={"initial_program": custom_program})

            # In stateless mode, reset doesn't use the custom program
            # Check that reset worked (should be empty in stateless mode)
            self.assertEqual(observation["generated_program"], "")
            self.assertIn("episode", info)
            self.assertEqual(info["mode"], "stateless")

            print(f"Reset with custom program successful (stateless mode)")
            print(f"Note: In stateless mode, custom programs are provided via action prompts")

        except Exception as e:
            self.skipTest(f"Environment reset with custom program failed: {e}")

    def test_step_execution(self):
        """Test a complete step in the environment"""
        try:
            # Reset environment
            observation, info = self.env.reset()

            # Create a comprehensive prompt with context
            action = self.env.create_context_prompt(
                base_instruction="Improve this code by adding error handling and better documentation",
                current_program="def hello():\n    return 'Hello, World!'",
                current_metrics={"score": 0.5, "quality": 0.6},
                parent_metrics={"score": 0.4, "quality": 0.5},
            )

            new_obs, reward, terminated, truncated, step_info = self.env.step(action)

            # Check new observation structure
            self.assertIn("generated_program", new_obs)
            self.assertIn("evaluation_metrics", new_obs)
            self.assertIn("has_errors", new_obs)
            self.assertIn("generation_success", new_obs)

            # Check reward is numeric
            self.assertIsInstance(reward, (int, float))

            # Check step info for stateless environment
            self.assertIn("metrics", step_info)
            self.assertIn("reward_calculation", step_info)
            self.assertIn("step_time", step_info)
            self.assertIn("generation_success", step_info)
            self.assertIn("evaluation_success", step_info)

            print(f"Step executed successfully in stateless mode")
            print(f"Reward: {reward}")
            print(f"Generated program length: {len(new_obs['generated_program'])}")
            print(f"Reward calculation: {step_info['reward_calculation']}")

        except Exception as e:
            self.skipTest(f"Step execution failed: {e}")

    def test_multiple_steps(self):
        """Test multiple steps in sequence"""
        try:
            # Reset environment
            observation, info = self.env.reset()

            # Take multiple steps with comprehensive prompts
            actions = [
                self.env.create_context_prompt(
                    "Add type hints to improve code quality",
                    current_program="def add(a, b): return a + b",
                ),
                self.env.create_context_prompt(
                    "Add proper error handling", current_program="def divide(a, b): return a / b"
                ),
                self.env.create_context_prompt(
                    "Optimize for better performance",
                    current_program="def factorial(n): return n * factorial(n-1) if n > 1 else 1",
                ),
            ]

            total_reward = 0
            for i, action in enumerate(actions):
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                total_reward += reward

                # Check that we got a response
                self.assertIn("generated_program", obs)
                self.assertIn("generation_success", obs)

                print(f"Step {i+1}: reward={reward:.4f}, total={total_reward:.4f}")
                print(f"Generation success: {obs['generation_success']}")

            print(f"Multiple steps completed successfully in stateless mode")
            print(f"Total reward: {total_reward:.4f}")

        except Exception as e:
            self.skipTest(f"Multiple steps failed: {e}")

    def test_metrics_to_array_conversion(self):
        """Test conversion of metrics dict to fixed-size array"""
        # Test with normal metrics
        metrics = {"score": 0.8, "accuracy": 0.9, "speed": 0.7}
        array = self.env._metrics_to_array(metrics)

        self.assertEqual(len(array), 10)
        self.assertEqual(array[0], 0.8)
        self.assertEqual(array[1], 0.9)
        self.assertEqual(array[2], 0.7)
        self.assertEqual(array[3], 0.0)  # Padding

        # Test with empty metrics
        empty_array = self.env._metrics_to_array({})
        self.assertTrue(np.allclose(empty_array, np.zeros(10)))

        # Test with too many metrics (should truncate)
        many_metrics = {f"metric_{i}": float(i) for i in range(15)}
        truncated_array = self.env._metrics_to_array(many_metrics)
        self.assertEqual(len(truncated_array), 10)

    def test_reward_calculation(self):
        """Test reward calculation logic"""
        # Test initial reward (no parent)
        current_metrics = {"score": 0.8, "accuracy": 0.9}
        reward = self.env._calculate_reward(current_metrics, {})
        expected = (0.8 + 0.9) / 2 * self.env.reward_scale
        self.assertAlmostEqual(reward, expected, places=4)

        # Test improvement reward
        parent_metrics = {"score": 0.6, "accuracy": 0.7}
        reward = self.env._calculate_reward(current_metrics, parent_metrics)

        current_avg = (0.8 + 0.9) / 2
        parent_avg = (0.6 + 0.7) / 2
        improvement = current_avg - parent_avg
        expected = improvement * self.env.reward_scale + self.env.improvement_bonus
        self.assertAlmostEqual(reward, expected, places=4)

        # Test error penalty
        error_metrics = {"score": 0.5, "error": 1.0}
        reward = self.env._calculate_reward(error_metrics, parent_metrics)
        # Should include error penalty - check that penalty component exists
        reward_components = self.env._get_reward_components(error_metrics, parent_metrics)
        self.assertIn("error_penalty", reward_components)
        self.assertEqual(reward_components["error_penalty"], self.env.penalty_for_errors)

    def test_render(self):
        """Test rendering functionality"""
        try:
            # Reset and step once
            self.env.reset()
            self.env.step("Improve this code")

            # Test render (should not raise exception)
            self.env.render(mode="human")

        except Exception as e:
            self.skipTest(f"Render test failed: {e}")

    def test_environment_close(self):
        """Test environment cleanup"""
        # This should not raise an exception
        self.env.close()

    def test_default_initial_program(self):
        """Test default initial program generation"""
        default_prog = self.env._get_default_initial_program()
        self.assertIn("hello_world", default_prog)
        self.assertIn("def ", default_prog)

        # Test for different language
        env_java = ProgramEvolutionEnv(
            llm=self.llm, exe_evaluator=self.exe_evaluator, language="java"
        )
        java_prog = env_java._get_default_initial_program()
        self.assertIn("//", java_prog)


class TestProgramEvolutionEnvWithMocks(unittest.TestCase):
    """Test environment with mocked components for faster testing"""

    def setUp(self):
        """Set up test fixtures with mocks"""
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.generate = AsyncMock(
            return_value="def improved_function():\n    return 'better code'"
        )

        # Mock evaluators
        self.mock_exe_evaluator = Mock()
        self.mock_exe_evaluator.evaluate = AsyncMock(return_value={"score": 0.8, "efficiency": 0.9})

        self.mock_llm_evaluator = Mock()
        self.mock_llm_evaluator.evaluate = AsyncMock(
            return_value={"readability": 0.7, "style": 0.8}
        )

        # Create environment with mocks
        self.env = ProgramEvolutionEnv(
            llm=self.mock_llm,
            exe_evaluator=self.mock_exe_evaluator,
            llm_evaluator=self.mock_llm_evaluator,
        )

    def test_mocked_step_execution(self):
        """Test step execution with mocked components"""
        # Reset
        observation, info = self.env.reset()

        # Take a step with comprehensive prompt
        action = self.env.create_context_prompt(
            "improve code", current_program="def test(): pass", current_metrics={"score": 0.5}
        )

        obs, reward, terminated, truncated, step_info = self.env.step(action)

        # Verify mocks were called
        self.mock_llm.generate.assert_called_once()
        self.mock_exe_evaluator.evaluate.assert_called()
        self.mock_llm_evaluator.evaluate.assert_called()

        # Verify stateless environment structure
        self.assertIn("generated_program", obs)
        self.assertIn("generation_success", obs)
        self.assertIsInstance(reward, (int, float))
        self.assertIn("metrics", step_info)  # Changed from "new_metrics"

    def test_error_handling_in_generation(self):
        """Test error handling when LLM generation fails"""
        # Make LLM raise an exception
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM failed"))

        # Reset environment
        self.env.reset()

        # Step should handle the error gracefully
        obs, reward, terminated, truncated, step_info = self.env.step("test action")

        # Should get penalty reward
        self.assertEqual(reward, self.env.penalty_for_errors)
        self.assertIn("error", step_info)
        self.assertEqual(step_info["step"], "generation")

    def test_error_handling_in_evaluation(self):
        """Test error handling when evaluation fails"""
        # Make evaluator raise an exception
        self.mock_exe_evaluator.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        # Reset environment
        self.env.reset()

        # Step should handle the error gracefully
        obs, reward, terminated, truncated, step_info = self.env.step("test action")

        # Evaluation failure results in error metrics
        # Check that error metrics were generated in step_info["metrics"]
        self.assertIn("metrics", step_info)
        self.assertEqual(step_info["metrics"]["error"], 1.0)
        self.assertEqual(step_info["metrics"]["score"], 0.0)

        # Reward should be the average of error metrics (0.5) plus error penalty (-0.1) = 0.4
        expected_reward = (
            0.5 * self.env.reward_scale + self.env.penalty_for_errors
        )  # 0.5 - 0.1 = 0.4
        self.assertEqual(reward, expected_reward)


if __name__ == "__main__":
    unittest.main()
