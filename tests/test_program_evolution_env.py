"""
Unit tests for ProgramEvolutionEnv

Tests the refactored environment with explicit Result pattern and common actions module.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from openevolve.environment.program_evolution_env import ProgramEvolutionEnv
from openevolve.common.actions import EvolutionAction


class TestProgramEvolutionEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.generate = AsyncMock(
            return_value="def improved_function():\n    return 'optimized'"
        )

        # Mock execution evaluator
        self.mock_exe_evaluator = MagicMock()
        self.mock_exe_evaluator.evaluate = AsyncMock(
            return_value={"score": 0.85, "execution_time": 0.1, "memory_usage": 1024}
        )

        # Mock LLM evaluator (optional)
        self.mock_llm_evaluator = MagicMock()
        self.mock_llm_evaluator.evaluate = AsyncMock(
            return_value={"code_quality": 0.9, "readability": 0.8}
        )

        # Mock reward extractor
        def mock_reward_extractor(info):
            return info.get("raw_metrics", {}).get("score", 0.0)

        # Create environment
        self.env = ProgramEvolutionEnv(
            llm=self.mock_llm,
            exe_evaluator=self.mock_exe_evaluator,
            llm_evaluator=self.mock_llm_evaluator,
            reward_extractor=mock_reward_extractor,
            language="python",
        )

    def test_environment_initialization(self):
        """Test environment is properly initialized"""
        self.assertIsNotNone(self.env.llm)
        self.assertIsNotNone(self.env.exe_evaluator)
        self.assertIsNotNone(self.env.llm_evaluator)
        self.assertIsNotNone(self.env.prompt_builder)
        self.assertIsNotNone(self.env.code_generator)
        self.assertEqual(self.env.language, "python")
        self.assertEqual(self.env.default_mode, "full_rewrite")

    def test_action_and_observation_spaces(self):
        """Test gymnasium spaces are properly defined"""
        from gymnasium.spaces import Dict

        # Check action space
        self.assertIsInstance(self.env.action_space, Dict)
        action_space = self.env.action_space  # Cast to Dict for type checking
        self.assertIn("instruction", action_space.spaces)  # type: ignore
        self.assertIn("mode", action_space.spaces)  # type: ignore

        # Check observation space
        self.assertIsInstance(self.env.observation_space, Dict)
        obs_space = self.env.observation_space  # Cast to Dict for type checking
        self.assertIn("generated_program", obs_space.spaces)  # type: ignore
        self.assertIn("evaluation_metrics", obs_space.spaces)  # type: ignore
        self.assertIn("success", obs_space.spaces)  # type: ignore

    def test_reset_functionality(self):
        """Test reset returns proper initial state"""
        obs, info = self.env.reset()

        self.assertIsInstance(obs, dict)
        self.assertIn("generated_program", obs)
        self.assertIn("evaluation_metrics", obs)
        self.assertIn("success", obs)

        self.assertEqual(obs["generated_program"], "")
        self.assertEqual(obs["success"], 0)
        self.assertIsInstance(obs["evaluation_metrics"], np.ndarray)

        self.assertIsInstance(info, dict)
        self.assertIn("episode", info)

    def test_step_with_evolution_action(self):
        """Test step with EvolutionAction object"""
        action = EvolutionAction(
            instruction="Optimize this function for better performance",
            current_program="def slow_function():\n    return sum(range(1000))",
            current_score=0.5,
            mode="full_rewrite",
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check observation
        self.assertIsInstance(obs, dict)
        self.assertIn("generated_program", obs)
        self.assertIn("evaluation_metrics", obs)
        self.assertIn("success", obs)
        self.assertEqual(obs["success"], 1)  # Should succeed

        # Check reward
        self.assertIsInstance(reward, (int, float))

        # Check termination flags
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

        # Check info
        self.assertIsInstance(info, dict)
        self.assertIn("raw_metrics", info)
        self.assertIn("generation_success", info)
        self.assertIn("evaluation_success", info)
        self.assertIn("evolution_action", info)
        self.assertIn("system_prompt", info)
        self.assertIn("user_prompt", info)

    def test_step_with_dict_action(self):
        """Test step with dictionary action"""
        action = {
            "instruction": "Improve code readability",
            "current_program": "def x(a,b):return a+b",
            "current_score": 0.3,
            "mode": "full_rewrite",
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs["success"], 1)
        self.assertIn("evolution_action", info)
        self.assertEqual(info["evolution_action"]["instruction"], "Improve code readability")

    def test_step_with_string_action(self):
        """Test step with string action (fallback)"""
        action = "Optimize this code for better performance"

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs["success"], 1)
        self.assertIn("evolution_action", info)
        self.assertEqual(info["evolution_action"]["instruction"], action)
        self.assertEqual(info["evolution_action"]["mode"], "full_rewrite")

    def test_process_action_method(self):
        """Test _process_action helper method"""
        # Test with EvolutionAction
        action = EvolutionAction(instruction="test", mode="diff")
        result = self.env._process_action(action)
        self.assertIsInstance(result, EvolutionAction)
        if result:
            self.assertEqual(result.instruction, "test")

        # Test with dict
        action_dict = {"instruction": "test dict", "mode": "full_rewrite"}
        result = self.env._process_action(action_dict)
        self.assertIsInstance(result, EvolutionAction)
        if result:
            self.assertEqual(result.instruction, "test dict")

        # Test with string
        action_str = "test string instruction"
        result = self.env._process_action(action_str)
        self.assertIsInstance(result, EvolutionAction)
        if result:
            self.assertEqual(result.instruction, action_str)

        # Test with invalid action - use wrong type
        result = self.env._process_action(123)  # type: ignore
        self.assertIsNone(result)

    def test_build_prompts_method(self):
        """Test _build_prompts helper method"""
        action = EvolutionAction(
            instruction="Test prompt building",
            current_program="def test(): pass",
            mode="full_rewrite",
        )

        result = self.env._build_prompts(action)
        self.assertIsNotNone(result)
        if result:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

            system_prompt, user_prompt = result
            self.assertIsInstance(system_prompt, str)
            self.assertIsInstance(user_prompt, str)
            self.assertIn("Test prompt building", user_prompt)

    def test_generate_code_method(self):
        """Test _generate_code helper method"""
        system_prompt = "You are a helpful coding assistant."
        user_prompt = "Improve this code: def add(a, b): return a + b"

        result = self.env._generate_code(system_prompt, user_prompt)

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("program", result)

        self.assertTrue(result["success"])
        self.assertEqual(result["error"], "")
        self.assertIsInstance(result["program"], str)
        self.assertTrue(len(result["program"]) > 0)

    def test_evaluate_code_safe_method(self):
        """Test _evaluate_code_safe helper method"""
        test_code = "def improved_function():\n    return 'test'"

        result = self.env._evaluate_code_safe(test_code)

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("metrics", result)

        self.assertTrue(result["success"])
        self.assertEqual(result["error"], "")
        self.assertIsInstance(result["metrics"], dict)

    def test_error_handling_invalid_action(self):
        """Test error handling with invalid action"""
        # Test with an invalid action type (using type: ignore to test error handling)
        obs, reward, terminated, truncated, info = self.env.step(123)  # type: ignore

        self.assertEqual(obs["success"], 0)
        self.assertIn("error", info)
        self.assertIn("Invalid action", info["error"])

    def test_error_handling_generation_failure(self):
        """Test error handling when code generation fails"""
        # Mock LLM to fail
        self.env.code_generator.llm.generate = AsyncMock(side_effect=Exception("LLM failed"))

        action = EvolutionAction(instruction="test", mode="full_rewrite")
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs["success"], 0)
        self.assertIn("error", info)
        self.assertIn("Code generation failed", info["error"])

    def test_error_handling_evaluation_failure(self):
        """Test error handling when evaluation fails"""
        # Mock evaluator to fail
        self.env.exe_evaluator.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        action = EvolutionAction(instruction="test", mode="full_rewrite")
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs["success"], 0)
        self.assertIn("error", info)
        self.assertIn("Code evaluation failed", info["error"])

    def test_metrics_cleaning(self):
        """Test metrics cleaning functionality"""
        dirty_metrics = {
            "score": "0.85",  # String number
            "time": 0.1,  # Float
            "invalid": "not_a_number",  # Invalid
            "count": 42,  # Integer
        }

        clean = self.env._clean_metrics(dirty_metrics)

        self.assertEqual(clean["score"], 0.85)
        self.assertEqual(clean["time"], 0.1)
        self.assertEqual(clean["invalid"], 0.0)  # Should default to 0.0
        self.assertEqual(clean["count"], 42.0)

    def test_metrics_to_array(self):
        """Test metrics to array conversion"""
        metrics = {"score": 0.85, "time": 0.1, "memory": 1024.0}
        array = self.env._to_array(metrics)

        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (10,))
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(array[0], 0.85)
        self.assertEqual(array[1], 0.1)
        self.assertEqual(array[2], 1024.0)

    def test_default_reward_extractor(self):
        """Test default reward extraction"""
        info = {"raw_metrics": {"score": 0.8, "time": 0.2}}

        reward = self.env._default_reward(info)
        self.assertEqual(reward, 0.5)  # Average of 0.8 and 0.2

    def test_render_and_close(self):
        """Test render and close methods don't crash"""
        # These should not raise exceptions
        self.env.render()
        self.env.close()


if __name__ == "__main__":
    unittest.main()
