"""Tests for program_evolution_env.py"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from openevolve.environment.program_evolution_env import ProgramEvolutionEnv


class TestProgramEvolutionEnv(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.generate = AsyncMock(return_value="def test():\n    return 'improved'")

        # Mock evaluator
        self.mock_evaluator = MagicMock()
        self.mock_evaluator.evaluate = AsyncMock(
            return_value={"score": 0.85, "execution_time": 0.1}
        )

        # Mock reward extractor
        def mock_reward_extractor(info):
            return info.get("raw_metrics", {}).get("score", 0.0)

        # Create environment
        self.env = ProgramEvolutionEnv(
            llm=self.mock_llm,
            exe_evaluator=self.mock_evaluator,
            llm_evaluator=None,
            reward_extractor=mock_reward_extractor,
        )

    def test_init(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env.llm)
        self.assertIsNotNone(self.env.exe_evaluator)
        self.assertIsNone(self.env.llm_evaluator)
        self.assertIsNotNone(self.env.reward_extractor)

    def test_action_space(self):
        """Test action space is Text"""
        from gymnasium.spaces import Text

        self.assertIsInstance(self.env.action_space, Text)

    def test_observation_space(self):
        """Test observation space is Dict"""
        from gymnasium.spaces import Dict

        self.assertIsInstance(self.env.observation_space, Dict)

    def test_create_prompt(self):
        """Test the static create_prompt method"""
        prompt = ProgramEvolutionEnv.create_prompt(
            instruction="Fix the bug",
            current_program="def add(a, b):\n    return a - b",
            current_score=0.5,
        )

        self.assertIn("Fix the bug", prompt)
        self.assertIn("def add(a, b):", prompt)
        self.assertIn("Current Score: 0.5", prompt)

    def test_step_basic(self):
        """Test basic step functionality"""
        # Create a simple prompt
        action = "Improve this function: def add(a, b): return a + b"

        # Run step
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Verify calls
        self.mock_llm.generate.assert_called()
        self.mock_evaluator.evaluate.assert_called()

        # Verify outputs
        self.assertIsInstance(observation, dict)
        self.assertIn("generated_program", observation)
        self.assertIn("evaluation_metrics", observation)
        self.assertIn("success", observation)

        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_default_reward_extractor(self):
        """Test the default reward extractor"""
        env = ProgramEvolutionEnv(llm=self.mock_llm, exe_evaluator=self.mock_evaluator)

        # Test with score in raw_metrics (average of 0.7 and 0.3 = 0.5)
        info = {"raw_metrics": {"score": 0.7, "other": 0.3}}
        reward = env.reward_extractor(info)
        self.assertEqual(reward, 0.5)

        # Test without score (only has 'other': 0.3)
        info = {"raw_metrics": {"other": 0.3}}
        reward = env.reward_extractor(info)
        self.assertEqual(reward, 0.3)

    def test_reset(self):
        """Test reset functionality"""
        observation, info = self.env.reset()

        self.assertIsInstance(observation, dict)
        self.assertIn("generated_program", observation)
        self.assertIn("evaluation_metrics", observation)
        self.assertIn("success", observation)

        self.assertIsInstance(info, dict)

    def test_render(self):
        """Test rendering functionality"""
        # Should not raise any errors
        result = self.env.render()
        self.assertIsNone(result)

    def test_close(self):
        """Test close functionality"""
        # Should not raise any errors
        self.env.close()


if __name__ == "__main__":
    unittest.main()
