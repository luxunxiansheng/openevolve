"""
Unit tests for ProgramEvolutionEnv

Tests the refactored environment with explicit Result pattern and common actions module.
"""

import unittest
import numpy as np

from opencontext.environment.program_evolution_env import ProgramEvolutionEnv
from opencontext.common.actions import EvolutionAction
from opencontext.llm.llm_openai import OpenAILLM
from opencontext.environment.evaluators.execution_evaluator import ExecutionEvaluator
from opencontext.environment.evaluators.llm_evaluator import LLMEvaluator


class TestProgramEvolutionEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Use real LLM and evaluators
        llm = OpenAILLM(
            api_base="http://localhost:8010/v1",  # or your real endpoint
            api_key="none",  # or your real key
            name="Qwen3-14B-AWQ",  # or your real model name
        )
        critic_path = "/workspaces/opencontext/examples/circle_packing_with_artifacts_new/critic.py"
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
        self.assertIsNotNone(self.env.evolution_llm)
        self.assertEqual(self.env.language, "python")
        self.assertEqual(self.env.default_mode, "full_rewrite")

    def test_action_and_observation_spaces(self):
        """Test gymnasium spaces are properly defined"""
        from gymnasium.spaces import Dict

        # Check action space - simplified for EvolutionAction only
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

        # Basic shape checks
        self.assertIsInstance(obs, dict)
        self.assertIn("generated_program", obs)
        self.assertIn("evaluation_metrics", obs)
        self.assertIn("success", obs)

        # success can be 0 or 1 depending on external services (LLM/Ray); be tolerant
        self.assertIn(obs["success"], (0, 1))

        # If success==1, verify the usual happy-path fields
        if obs["success"] == 1:
            # Check reward
            self.assertIsInstance(reward, (int, float))

            # Check termination flags
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)

            # Check info content
            self.assertIsInstance(info, dict)
            self.assertIn("raw_metrics", info)
            self.assertIn("generation_success", info)
            self.assertIn("evaluation_success", info)
            self.assertIn("evolution_action", info)
            self.assertIn("system_prompt", info)
            self.assertIn("user_prompt", info)
        else:
            # Failure path -- ensure the environment reports an error and didn't crash
            self.assertIsInstance(info, dict)
            self.assertIn("error", info)
            # reward should still be numeric
            self.assertIsInstance(reward, (int, float))

    def test_step_with_dict_action(self):
        """Test step with dictionary action - should fail as only EvolutionAction is allowed"""
        action = {
            "instruction": "Improve code readability",
            "current_program": "def x(a,b):return a+b",
            "current_score": 0.3,
            "mode": "full_rewrite",
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Should fail since dict is not allowed
        self.assertEqual(obs["success"], 0)
        self.assertIn("error", info)
        self.assertIn("Invalid action type", info["error"])

    def test_error_handling_invalid_action(self):
        """Test error handling with invalid action"""
        # Test with an invalid action type (using type: ignore to test error handling)
        obs, reward, terminated, truncated, info = self.env.step(123)  # type: ignore

        self.assertEqual(obs["success"], 0)
        self.assertIn("error", info)
        self.assertIn("Invalid action type", info["error"])

    def test_error_handling_string_action(self):
        """Test that string actions are rejected"""
        action = "Optimize this code for better performance"

        obs, reward, terminated, truncated, info = self.env.step(action)  # type: ignore

        self.assertEqual(obs["success"], 0)
        self.assertIn("error", info)
        self.assertIn("Invalid action type", info["error"])

    # The following error-handling tests previously used AsyncMock to simulate LLM/evaluator failure.
    # Since only real components are allowed, and we cannot force real LLM/evaluator to fail here,
    # these tests are removed. If you want to test real failure, use a real endpoint/critic that fails or add such a test in a real scenario.

    def test_render_and_close(self):
        """Test render and close methods don't crash"""
        # These should not raise exceptions
        self.env.render()
        self.env.close()


if __name__ == "__main__":
    unittest.main()
