"""
Simple example of EvolutionPromptBuilder using existing sampler and templates
"""

from opencontext.prompt.prompt_builder import EvolutionPromptBuilder


def main():
    """Demonstrate the simplified EvolutionPromptBuilder"""

    # Initialize the prompt builder
    builder = EvolutionPromptBuilder(
        evolution_mode="full_rewrite",  # or "diff"
        num_top_programs=2,
        num_diverse_programs=2,
        include_artifacts=True,
    )

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

    # Current performance metrics
    current_metrics = {"speed": 0.3, "correctness": 1.0, "memory_efficiency": 0.6}

    # Example top-performing programs
    top_programs = [
        {
            "code": "def quicksort(arr): return sorted(arr)",
            "metrics": {"speed": 0.95, "correctness": 1.0, "memory_efficiency": 0.9},
        },
        {
            "code": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
""",
            "metrics": {"speed": 0.85, "correctness": 1.0, "memory_efficiency": 0.8},
        },
    ]

    # Example diverse programs for inspiration
    diverse_programs = [
        {
            "code": """
import heapq
def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]
""",
            "metrics": {"speed": 0.75, "correctness": 1.0, "memory_efficiency": 0.7},
        }
    ]

    # Example execution artifacts
    execution_artifacts = {
        "stderr": "",
        "performance_profile": "95% time in nested loops",
        "memory_usage": "Peak: 1.2MB for 1000 elements",
    }

    # Build evolution prompt
    prompt = builder.build_evolution_prompt(
        current_program=current_program,
        current_metrics=current_metrics,
        top_programs=top_programs,
        diverse_programs=diverse_programs,
        language="python",
        problem_description="Optimize sorting algorithm for better performance",
        execution_artifacts=execution_artifacts,
    )

    print("=== EVOLUTION PROMPT ===")
    print("System:", prompt["system"][:200] + "...")
    print("\nUser prompt length:", len(prompt["user"]))
    print("User prompt preview:", prompt["user"][:500] + "...")

    # Example critic prompt
    critic_prompt = builder.build_critic_prompt(
        program_to_evaluate=current_program, language="python"
    )

    print("\n=== CRITIC PROMPT ===")
    print("System:", critic_prompt["system"][:200] + "...")
    print("User:", critic_prompt["user"][:300] + "...")

    # Example crossover prompt
    parent_a = top_programs[0]
    parent_b = diverse_programs[0]

    crossover_prompt = builder.build_crossover_prompt(
        parent_a=parent_a, parent_b=parent_b, language="python"
    )

    print("\n=== CROSSOVER PROMPT ===")
    print("System:", crossover_prompt["system"][:200] + "...")
    print("User preview:", crossover_prompt["user"][:300] + "...")


if __name__ == "__main__":
    main()
