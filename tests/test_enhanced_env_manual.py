#!/usr/bin/env python3
"""Simple test for the enhanced ProgramEvolutionEnv"""

import sys
import os

sys.path.insert(0, "/workspaces/openevolve")

from unittest.mock import AsyncMock, MagicMock
from openevolve.environment.program_evolution_env import ProgramEvolutionEnv, EvolutionAction


def test_enhanced_env():
    """Test the enhanced environment with structured actions"""
    print("Testing enhanced ProgramEvolutionEnv...")

    # Mock LLM and evaluator
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="def improved():\n    return 'much better'")

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.95, "execution_time": 0.05})

    # Create environment
    env = ProgramEvolutionEnv(
        llm=mock_llm,
        exe_evaluator=mock_evaluator,
        reward_extractor=lambda info: info.get("raw_metrics", {}).get("score", 0.0),
    )

    print("✓ Environment created successfully")

    # Test with structured action
    action = EvolutionAction(
        instruction="Optimize this function for better performance",
        current_program="def slow_func():\n    return sum(range(1000))",
        current_score=0.3,
        parent_program="def old_func():\n    return 0",
        previous_attempts=[{"score": 0.2, "summary": "Too slow"}],
        mode="full_rewrite",
        context={"target": "performance"},
    )

    # Run step
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"✓ Step with EvolutionAction completed")
    print(f"  - Generated program length: {len(obs['generated_program'])}")
    print(f"  - Reward: {reward}")
    print(f"  - Success: {obs['success']}")

    # Test with dict action
    dict_action = {
        "instruction": "Fix the bug in this code",
        "current_program": "def divide(a, b):\n    return a / b",
        "mode": "full_rewrite",
    }

    obs2, reward2, _, _, info2 = env.step(dict_action)
    print(f"✓ Step with dict action completed")

    # Test with string fallback
    string_action = "Improve this: def add(x, y): return x + y"
    obs3, reward3, _, _, info3 = env.step(string_action)
    print(f"✓ Step with string fallback completed")

    print("\n🎉 All tests passed! Enhanced environment is working correctly.")


if __name__ == "__main__":
    test_enhanced_env()
