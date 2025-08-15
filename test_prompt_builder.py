"""
Test EvolutionPromptBuilder directly
"""

import sys
import os

sys.path.append("/workspaces/opencontext")

from opencontext.prompt.prompt_builder import EvolutionPromptBuilder


def test_prompt_builder():
    """Test the EvolutionPromptBuilder directly"""

    try:
        # Initialize the builder
        builder = EvolutionPromptBuilder()

        # Example program to evolve
        current_program = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

        # Example elite programs
        elite_programs = [
            {
                "code": "def quick_sort(arr): return sorted(arr)",
                "metrics": {"speed": 0.95, "correctness": 1.0},
            }
        ]

        # Current metrics
        current_metrics = {"speed": 0.3, "correctness": 1.0}

        # Test diff-based evolution prompt
        print("=== Testing EvolutionPromptBuilder ===")
        diff_prompt = builder.build_diff_evolution_prompt(
            current_program=current_program,
            elite_programs=elite_programs,
            current_metrics=current_metrics,
            language="python",
        )

        print("✓ Successfully built diff evolution prompt")
        print(f"System message length: {len(diff_prompt['system'])} chars")
        print(f"User message length: {len(diff_prompt['user'])} chars")

        # Test rewrite evolution prompt
        rewrite_prompt = builder.build_rewrite_evolution_prompt(
            current_program=current_program,
            elite_programs=elite_programs,
            current_metrics=current_metrics,
        )

        print("✓ Successfully built rewrite evolution prompt")
        print(f"System message length: {len(rewrite_prompt['system'])} chars")
        print(f"User message length: {len(rewrite_prompt['user'])} chars")

        # Test evaluation prompt
        eval_prompt = builder.build_evaluation_prompt(program=current_program)

        print("✓ Successfully built evaluation prompt")
        print(f"System message length: {len(eval_prompt['system'])} chars")
        print(f"User message length: {len(eval_prompt['user'])} chars")

        print("\n=== EvolutionPromptBuilder Test PASSED ===")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_prompt_builder()
    if success:
        print("\nThe EvolutionPromptBuilder is working correctly!")
        print("It uses the existing PromptSampler and Templates infrastructure")
        print("to create evolution-focused prompts for LLM program generation.")
    else:
        print("\nTest failed - there may be import or configuration issues.")
