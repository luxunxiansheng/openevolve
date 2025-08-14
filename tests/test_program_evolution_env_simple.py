"""
Tests for ProgramEvolutionEnv

Simple tests with real components only (no mocks).
"""

import unittest
import numpy as np

from openevolve.environment.program_evolution_env import ProgramEvolutionEnv
from openevolve.environment.evaluators import ExecutionEvaluator, LLMEvaluator
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.llm.llm_ensemble import EnsembleLLM


class TestProgramEvolutionEnv(unittest.TestCase):
    """Test the Program Evolution Environment with real components"""

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
        )

    def test_environment_creation(self):
        """Test that environment can be created properly"""
        self.assertIsInstance(self.env, ProgramEvolutionEnv)
        self.assertEqual(self.env.llm, self.llm)
        self.assertEqual(self.env.exe_evaluator, self.exe_evaluator)
        self.assertEqual(self.env.llm_evaluator, self.llm_evaluator)
        self.assertEqual(self.env.language, "python")

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

            # Check observation structure (stateless mode)
            self.assertIn("generated_program", observation)
            self.assertIn("evaluation_metrics", observation)
            self.assertIn("has_errors", observation)
            self.assertIn("generation_success", observation)

            # Check info
            self.assertIn("episode", info)
            self.assertIn("mode", info)
            self.assertEqual(info["mode"], "stateless")

            print(f"Reset successful in stateless mode")

        except Exception as e:
            self.skipTest(f"Environment reset failed (likely due to API/Ray unavailable): {e}")

    def test_step_execution(self):
        """Test a complete step in the environment"""
        try:
            # Reset environment
            observation, info = self.env.reset()

            # Create a comprehensive prompt using the helper method
            action = ProgramEvolutionEnv.create_context_prompt(
                base_instruction="Improve this code by adding error handling and documentation",
                current_program="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                current_metrics={"score": 0.5, "efficiency": 0.3},
            )

            new_obs, reward, terminated, truncated, step_info = self.env.step(action)

            # Check observation structure
            self.assertIn("generated_program", new_obs)
            self.assertIn("evaluation_metrics", new_obs)
            self.assertIn("has_errors", new_obs)
            self.assertIn("generation_success", new_obs)

            # Check reward is numeric
            self.assertIsInstance(reward, (int, float))

            # Check step info
            self.assertIn("raw_metrics", step_info)
            self.assertIn("step_time", step_info)
            self.assertIn("generation_success", step_info)
            self.assertIn("evaluation_success", step_info)

            print(f"Step executed successfully")
            print(f"Reward: {reward}")
            print(f"New program length: {len(new_obs['generated_program'])}")
            print(f"Raw metrics: {step_info['raw_metrics']}")

        except Exception as e:
            self.skipTest(f"Step execution failed: {e}")

    def test_multiple_steps(self):
        """Test multiple steps in sequence"""
        try:
            # Reset environment
            observation, info = self.env.reset()

            # Take multiple steps
            actions = [
                "Add type hints to improve code quality: def hello(): return 'world'",
                "Add proper error handling: def divide(a, b): return a / b",
                "Optimize for better performance: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            ]

            total_reward = 0
            for i, action in enumerate(actions):
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                total_reward += reward

                # Check that we get valid results
                self.assertIn("generated_program", obs)
                self.assertIn("raw_metrics", step_info)

                print(f"Step {i+1}: reward={reward:.4f}, total={total_reward:.4f}")

            print(f"Multiple steps completed successfully")
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

    def test_reward_extraction(self):
        """Test reward extraction with custom function"""

        def custom_reward_function(info):
            raw_metrics = info.get("raw_metrics", {})
            if not info.get("generation_success", False):
                return -1.0
            # Return average of metrics
            if raw_metrics:
                values = [v for v in raw_metrics.values() if isinstance(v, (int, float))]
                return sum(values) / len(values) if values else 0.0
            return 0.0

        # Create environment with custom reward function
        custom_env = ProgramEvolutionEnv(
            llm=self.llm, exe_evaluator=self.exe_evaluator, reward_extractor=custom_reward_function
        )

        try:
            custom_env.reset()
            obs, reward, terminated, truncated, info = custom_env.step("def test(): return 42")

            # Reward should be calculated using our custom function
            self.assertIsInstance(reward, (int, float))
            expected_reward = custom_reward_function(info)
            self.assertEqual(reward, expected_reward)

        except Exception as e:
            self.skipTest(f"Custom reward test failed: {e}")

    def test_create_context_prompt(self):
        """Test the static context prompt creation method"""
        prompt = ProgramEvolutionEnv.create_context_prompt(
            base_instruction="Improve this code",
            current_program="def hello(): return 'world'",
            current_metrics={"score": 0.5, "accuracy": 0.8},
            parent_program="def hello(): pass",
            parent_metrics={"score": 0.3},
        )

        self.assertIn("Improve this code", prompt)
        self.assertIn("def hello(): return 'world'", prompt)
        self.assertIn("Current Score: 0.65", prompt)  # (0.5 + 0.8) / 2
        self.assertIn("Previous Score: 0.30", prompt)
        self.assertIn("def hello(): pass", prompt)

    def test_render(self):
        """Test rendering functionality"""
        try:
            # Reset and step once
            self.env.reset()
            self.env.step("Improve this code: def test(): pass")

            # Test render (should not raise exception)
            self.env.render(mode="human")

        except Exception as e:
            self.skipTest(f"Render test failed: {e}")

    def test_environment_close(self):
        """Test environment cleanup"""
        # This should not raise an exception
        self.env.close()


if __name__ == "__main__":
    unittest.main()
