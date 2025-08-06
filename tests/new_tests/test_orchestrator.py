#!/usr/bin/env python3
"""
Test script for the Orchestrator class to verify it works correctly with Ray actors.
This test uses real components and configurations, not mocks.
"""

import asyncio
import os
import tempfile
import time
import ray

# Import required classes
from openevolve.orchestration.orchestrator import Orchestrator
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.database.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler
from openevolve.critic.llm_critic import LLMCritic
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.critic.critic import EvaluationResult, Critic
from openevolve.llm.llm_ensemble import EnsembleLLM



class TestEvaluator(Critic):
    """A real evaluator that provides meaningful evaluation for testing."""

    def __init__(self):
        super().__init__()
        self.evaluation_count = 0

    async def evaluate(self, program_code: str = "", **kwargs) -> EvaluationResult:
        """
        Evaluate program based on actual criteria:
        - Code length (preference for concise code)
        - Syntax validity
        - Contains print statement (for hello world style)
        """
        self.evaluation_count += 1

        # Initialize scores
        combined_score = 0.0
        code_length_score = 0.0
        syntax_score = 0.0
        functionality_score = 0.0

        # Score based on code length (prefer moderate length)
        code_length = len(program_code)
        if 10 <= code_length <= 200:
            code_length_score = 10.0
        elif code_length < 10:
            code_length_score = code_length  # Too short
        else:
            code_length_score = max(0, 10.0 - (code_length - 200) / 100)  # Too long

        # Check syntax validity
        try:
            compile(program_code, "<string>", "exec")
            syntax_score = 10.0
        except SyntaxError:
            syntax_score = 0.0

        # Check for basic functionality (contains print)
        if "print" in program_code.lower():
            functionality_score = 10.0
        elif "hello" in program_code.lower() or "world" in program_code.lower():
            functionality_score = 5.0

        # Combine scores
        combined_score = code_length_score * 0.3 + syntax_score * 0.4 + functionality_score * 0.3

        # Add small evolution bonus
        combined_score += self.evaluation_count * 0.1

        metrics = {
            "combined_score": combined_score,
            "code_length_score": code_length_score,
            "syntax_score": syntax_score,
            "functionality_score": functionality_score,
            "code_length": code_length,
            "evaluation_count": self.evaluation_count,
        }

        artifacts = {
            "evaluation_log": f"Evaluation #{self.evaluation_count}",
            "program_analysis": {
                "has_print": "print" in program_code.lower(),
                "has_hello_world": any(word in program_code.lower() for word in ["hello", "world"]),
                "syntax_valid": syntax_score > 0,
            },
            "timestamp": time.time(),
        }

        self.log_artifact(artifacts)
        self.log_metrics(metrics)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)


async def setup_test_environment():
    """Set up the test environment with Ray actors and components."""

    print("🔧 Setting up test environment...")

    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="orchestrator_test_")
    print(f"📁 Test output directory: {output_dir}")

    # Load real configuration from default config
    config_path = "/workspaces/openevolve/configs/default_config.yaml"
    print(f"📋 Loading configuration from: {config_path}")
    config = Config.from_yaml(config_path)

    # Override some settings for testing
    config.max_iterations = 3  # Keep test short
    config.diff_based_evolution = False  # Simpler for testing
    config.max_code_length = 1000

    # Create database as a Ray actor
    print("🗄️ Creating database actor...")
    database_actor = ray.remote(ProgramDatabase).remote(config.database)

    # Create other components
    print("🔧 Creating prompt sampler...")
    prompt_sampler = PromptSampler(config.prompt)

    print("🤖 Creating LLM client...")
    llm_client = EnsembleLLM([config.llm])

    print("🧠 Creating critics...")
    llm_critic = LLMCritic(llm_client, prompt_sampler)

    # Create a custom execution critic that uses our test evaluator
    class TestExecutionCritic(PythonExecutionCritic):
        def __init__(self, test_evaluator):
            super().__init__()
            self.test_evaluator = test_evaluator

        async def evaluate(self, program_code: str = "", **kwargs) -> EvaluationResult:
            # Use our test evaluator instead of the default execution
            return await self.test_evaluator.evaluate(program_code=program_code, **kwargs)

    # Create test evaluator and execution critic
    print("📊 Creating evaluator...")
    test_evaluator = TestEvaluator()
    exe_critic = TestExecutionCritic(test_evaluator)

    # Create evolution actor
    print("🧬 Creating evolution actor...")
    evolution_actor = EvolutionActor(
        database=database_actor,  # type: ignore
        prompt_sampler=prompt_sampler,
        llm_actor_client=llm_client,
        llm_critic=llm_critic,
        exe_critic=exe_critic,
        language="python",
        iteration=1,
        diff_based_evolution=config.diff_based_evolution,
        max_code_length=config.max_code_length,
        use_llm_critic=False,  # Disable for simpler testing
        llm_feedback_weight=0.1,
        artifacts_enabled=True,
    )

    # Create initial test program
    initial_program = Program(
        id="initial_test_program",
        code='print("Hello, World!")\n# This is a test program for evolution',
        language="python",
        parent_id=None,
        generation=0,
        metrics={"combined_score": 5.0, "code_length": 50},
        iteration_found=0,
        metadata={"test": True},
    )

    # Create orchestrator
    print("🎭 Creating orchestrator...")
    orchestrator = Orchestrator(
        config=config,
        initial_program=initial_program,
        database=database_actor,
        evolution_actor=evolution_actor,
        output_dir=output_dir,
        target_score=15.0,  # Set a reasonable target for testing
        max_iterations=config.max_iterations,
        language="python",
        programs_per_island=2,  # Small island size for testing
    )

    return orchestrator, database_actor, output_dir


async def verify_database_state(database_actor):
    """Verify the database contains expected data."""

    print("🔍 Verifying database state...")

    # Get all programs
    all_programs = ray.get(database_actor.get_all_programs.remote())
    print(f"📊 Total programs in database: {len(all_programs)}")

    # Get best program
    best_program = ray.get(database_actor.get_best_program.remote())
    if best_program:
        print(f"🏆 Best program ID: {best_program.id}")  # type: ignore
        print(f"📈 Best program metrics: {best_program.metrics}")  # type: ignore
    else:
        print("❌ No best program found")

    # Check island status
    try:
        ray.get(database_actor.log_island_status.remote())
        print("🏝️ Island status logged successfully")
    except Exception as e:
        print(f"⚠️ Island status error: {e}")

    return len(all_programs), best_program


def verify_output_files(output_dir):
    """Verify that output files were created correctly."""

    print("📋 Verifying output files...")

    best_dir = os.path.join(output_dir, "best")
    if os.path.exists(best_dir):
        print(f"✅ Best directory exists: {best_dir}")

        # Check for best program file
        best_files = os.listdir(best_dir)
        print(f"📄 Files in best directory: {best_files}")

        # Check for expected files
        expected_files = ["best_program.py", "best_program_info.json"]
        for expected_file in expected_files:
            if expected_file in best_files:
                file_path = os.path.join(best_dir, expected_file)
                file_size = os.path.getsize(file_path)
                print(f"✅ {expected_file} exists ({file_size} bytes)")
            else:
                print(f"❌ {expected_file} missing")
    else:
        print(f"❌ Best directory does not exist: {best_dir}")

    return os.path.exists(best_dir)


async def run_orchestrator_test():
    """Main test function that runs the orchestrator and verifies results."""

    print("🚀 Starting Orchestrator Integration Test")
    print("=" * 50)

    try:
        # Setup test environment
        orchestrator, database_actor, output_dir = await setup_test_environment()

        print("\n🏃 Running orchestrator...")
        start_time = time.time()

        # Run the orchestrator
        await orchestrator.run()

        end_time = time.time()
        print(f"⏱️ Orchestrator completed in {end_time - start_time:.2f} seconds")

        # Verify results
        print("\n🧪 Verifying test results...")

        # Check database state
        program_count, best_program = await verify_database_state(database_actor)

        # Check output files
        files_created = verify_output_files(output_dir)

        # Test summary
        print("\n📋 Test Summary:")
        print("=" * 30)

        success_criteria = [
            (program_count > 1, f"Multiple programs created: {program_count}"),
            (best_program is not None, "Best program exists"),
            (files_created, "Output files created"),
        ]

        all_passed = True
        for passed, description in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {description}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 All tests PASSED! Orchestrator is working correctly.")
        else:
            print("\n💥 Some tests FAILED! Check the orchestrator implementation.")

        # Cleanup
        print(f"\n🧹 Test output saved in: {output_dir}")
        print("Note: Temporary files are preserved for inspection")

        return all_passed

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    print("Orchestrator Integration Test")
    print("This test verifies the orchestrator works correctly with Ray actors.")
    print()

    # Run the test
    success = asyncio.run(run_orchestrator_test())

    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"\nTest completed with exit code: {exit_code}")
    exit(exit_code)
