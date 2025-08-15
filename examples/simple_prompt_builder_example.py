"""
Simple example of using EvolutionPromptBuilder with existing infrastructure
"""

from opencontext.prompt.prompt_builder import EvolutionPromptBuilder


def test_evolution_prompt_builder():
    """Test the EvolutionPromptBuilder"""

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
    print("=== DIFF-BASED EVOLUTION PROMPT ===")
    diff_prompt = builder.build_diff_evolution_prompt(
        current_program=current_program,
        elite_programs=elite_programs,
        current_metrics=current_metrics,
        language="python",
    )

    print("SYSTEM:", diff_prompt["system"])
    print("\nUSER:", diff_prompt["user"])

    # Test rewrite evolution prompt
    print("\n\n=== REWRITE EVOLUTION PROMPT ===")
    rewrite_prompt = builder.build_rewrite_evolution_prompt(
        current_program=current_program,
        elite_programs=elite_programs,
        current_metrics=current_metrics,
        language="python",
    )

    print("SYSTEM:", rewrite_prompt["system"])
    print("\nUSER:", rewrite_prompt["user"])

    # Test evaluation prompt
    print("\n\n=== EVALUATION PROMPT ===")
    eval_prompt = builder.build_evaluation_prompt(program=current_program, language="python")

    print("SYSTEM:", eval_prompt["system"])
    print("\nUSER:", eval_prompt["user"])


if __name__ == "__main__":
    test_evolution_prompt_builder()
