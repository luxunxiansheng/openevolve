import unittest
import asyncio
import logging

from opencontext.orchestration.orchestrator import Orchestrator

# Import the Orchestrator and its config

python_critic_path = "/workspaces/opencontext/examples/circle_packing_with_artifacts_new/critic.py"  # Replace with an actual script path
evoved_program_path = "/workspaces/opencontext/examples/circle_packing_with_artifacts_new/circle_packing.py"  # Replace with an actual script path
output_path = "/workspaces/opencontext/tests/outputs"

# Set up logging
logging.basicConfig(level=logging.INFO)


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator(
            critic_program_path=python_critic_path,
            evoved_program_path=evoved_program_path,
            output_dir=output_path,
        )

    def tearDown(self):
        # Cleanup after tests
        pass

    def test_run_basic(self):
        # Test basic run of orchestrator
        async def run_test():
            await self.orchestrator.run()

        asyncio.run(run_test())

    def test_save_best_program(self):
        # Test saving best program logic
        pass

    # Add more test methods as needed


if __name__ == "__main__":
    unittest.main()
