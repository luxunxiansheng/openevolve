"""
Demo showing actual prompt output from EvolutionPromptBuilder
"""

import sys

sys.path.append("/workspaces/opencontext")

from opencontext.prompt.prompt_builder import EvolutionPromptBuilder


def demo_prompt_output():
    """Show actual prompt output"""

    # Initialize the builder
    builder = EvolutionPromptBuilder()

    # Example program to evolve
    current_program = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

    # Example elite programs for context
    elite_programs = [
        {
            "code": """
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
            "metrics": {"speed": 0.95, "memory": 0.90},
        }
    ]

    # Current metrics
    current_metrics = {"speed": 0.2, "memory": 0.1}

    # Build diff evolution prompt
    prompt = builder.build_diff_evolution_prompt(
        current_program=current_program,
        elite_programs=elite_programs,
        current_metrics=current_metrics,
        language="python",
    )

    print("=" * 60)
    print("EVOLUTION PROMPT EXAMPLE")
    print("=" * 60)

    print("\nSYSTEM MESSAGE:")
    print("-" * 40)
    print(prompt["system"])

    print("\nUSER MESSAGE:")
    print("-" * 40)
    print(prompt["user"])

    print("\n" + "=" * 60)
    print("This shows how EvolutionPromptBuilder uses:")
    print("1. Templates.ACTOR_SYSTEM for evolution-focused system message")
    print("2. Templates.DIFF_USER for SEARCH/REPLACE format instructions")
    print("3. Existing PromptSampler infrastructure for formatting")
    print("4. Elite programs as context for learning")
    print("5. Current metrics for improvement guidance")


if __name__ == "__main__":
    demo_prompt_output()
