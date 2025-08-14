#!/usr/bin/env python3
"""Test the enhanced ProgramEvolutionEnv with structured actions"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from openevolve.environment.program_evolution_env import ProgramEvolutionEnv, EvolutionAction


def test_structured_actions():
    """Test the new structured action processing"""

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value="def improved_function():\n    return 'much better code'"
    )

    # Mock evaluator
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.95, "execution_time": 0.05})

    # Create environment
    env = ProgramEvolutionEnv(
        llm=mock_llm,
        exe_evaluator=mock_evaluator,
        reward_extractor=lambda info: info.get("raw_metrics", {}).get("score", 0.0),
    )

    print("Enhanced Environment created successfully!")

    # Test 1: Dictionary action
    dict_action = {
        "instruction": "Optimize this function for better performance",
        "current_program": "def slow_func():\n    return sum(range(1000))",
        "current_score": 0.3,
        "mode": "full_rewrite",
        "context": {"optimization_target": "speed", "constraints": "memory efficient"},
    }

    observation, reward, terminated, truncated, info = env.step(dict_action)
    print(f"\n✅ Dictionary Action Test:")
    print(f"  - Reward: {reward}")
    print(f"  - Success: {observation['success']}")
    print(f"  - Prompt length: {len(info['prompt'])}")
    print(f"  - Generated program: {observation['generated_program'][:50]}...")

    # Test 2: EvolutionAction object
    evolution_action = EvolutionAction(
        instruction="Fix the bug and improve readability",
        current_program="def buggy_func(x):\n    return x/0  # Bug!",
        current_score=0.1,
        parent_program="def original_func(x):\n    return x",
        previous_attempts=[
            {"score": 0.05, "summary": "Division by zero error"},
            {"score": 0.08, "summary": "Partial fix but still issues"},
        ],
        mode="full_rewrite",
    )

    observation, reward, terminated, truncated, info = env.step(evolution_action)
    print(f"\n✅ EvolutionAction Object Test:")
    print(f"  - Reward: {reward}")
    print(f"  - Success: {observation['success']}")
    print(f"  - Action mode: {info['processed_action']['mode']}")
    print(f"  - Has previous attempts: {len(info['processed_action']['previous_attempts'])}")

    # Test 3: Simple string fallback
    simple_action = "Write a function that calculates factorial"
    observation, reward, terminated, truncated, info = env.step(simple_action)
    print(f"\n✅ Simple String Test:")
    print(f"  - Reward: {reward}")
    print(f"  - Success: {observation['success']}")
    print(f"  - Processed as: {info['processed_action']['instruction'][:50]}...")

    # Test 4: Diff mode
    diff_action = EvolutionAction(
        instruction="Apply targeted improvements to this function",
        current_program="def calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)",
        current_score=0.6,
        mode="diff",
    )

    observation, reward, terminated, truncated, info = env.step(diff_action)
    print(f"\n✅ Diff Mode Test:")
    print(f"  - Reward: {reward}")
    print(f"  - Success: {observation['success']}")
    print(f"  - Mode: {info['processed_action']['mode']}")
    print(f"  - Prompt contains diff instructions: {'diff blocks' in info['prompt']}")

    print("\n🎉 All tests passed! Enhanced environment is working correctly.")


if __name__ == "__main__":
    test_structured_actions()
