"""
Unit tests for ProgramEvolutionEnv

Tests the refactored environment with explicit Result pattern and common actions module.
"""

import unittest
import numpy as np

from openevolve.environment.program_evolution_env import ProgramEvolutionEnv
from openevolve.common.actions import EvolutionAction
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.environment.evaluators.execution_evaluator import ExecutionEvaluator
from openevolve.environment.evaluators.llm_evaluator import LLMEvaluator


class TestProgramEvolutionEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Use real LLM and evaluators, as in test_evaluators.py
        llm = OpenAILLM(
            api_base="http://localhost:8010/v1",  # or your real endpoint
            api_key="none",  # or your real key
            name="Qwen3-14B-AWQ",  # or your real model name
        )
        critic_path = "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py"
        exe_evaluator = ExecutionEvaluator(critic_program_path=critic_path, job_timeout_seconds=10)
        llm_evaluator = LLMEvaluator(llm)
        self.env = ProgramEvolutionEnv(
            llm=llm, exe_evaluator=exe_evaluator, llm_evaluator=llm_evaluator, language="python"
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

        # For string actions, either success with evolution_action in info, or failure with error
        if obs["success"] == 1:
            self.assertIn("evolution_action", info)
            self.assertEqual(info["evolution_action"]["instruction"], action)
            self.assertEqual(info["evolution_action"]["mode"], "full_rewrite")
        else:
            # In case of failure, check that error is reported properly
            self.assertIn("error", info)
            self.assertIn("generation_success", info)
            self.assertIn("evaluation_success", info)

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

    # The following error-handling tests previously used AsyncMock to simulate LLM/evaluator failure.
    # Since only real components are allowed, and we cannot force real LLM/evaluator to fail here,
    # these tests are removed. If you want to test real failure, use a real endpoint/critic that fails or add such a test in a real scenario.

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
