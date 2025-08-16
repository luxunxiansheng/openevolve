import asyncio
import os
import unittest
import logging
import sys

from opencontext.common.actions import EvolutionAction, EvolutionMode
from opencontext.common.program import Program
from opencontext.environment.program_evolution import ProgramEvolutionEngine
from opencontext.environment import setup_environment_logging
from opencontext.llm.llm_openai import OpenAILLM

# Set up logging at module level so it's available for all tests
setup_environment_logging(level="DEBUG", include_extra=True)

# Also ensure unittest output doesn't interfere
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)


class TestProgramEvolutionEngineIntegration(unittest.TestCase):
    # Use default OpenAILLM settings (defaults point to localhost-compatible endpoint)
    # Environment variables may override api_base, api_key and model when available.

    def setUp(self):
        # Create logger for the test
        self.logger = logging.getLogger("test.program_evolution_engine")
        self.logger.info("=" * 50)
        self.logger.info("Setting up test environment")

        self.llm = OpenAILLM()
        # Pass logger to engine to see detailed logging
        self.engine = ProgramEvolutionEngine(self.llm, logger=self.logger)
        self.logger.info("Test setup completed")
        self.logger.info("=" * 50)

    def test_generate_code_async_full_rewrite(self):
        self.logger.info("Starting test_generate_code_async_full_rewrite")

        current_program = Program(
            id="test-1", code="def add(a, b):\n    return a+b", language="python"
        )

        action = EvolutionAction(
            goal="Add a docstring to the function",
            instructions=["Add a docstring to the function in Chinese"],
            current_program=current_program,
            mode=EvolutionMode.FULL_REWRITE,
        )

        self.logger.info("Executing code generation test")
        # Run the async generation; if it fails the exception will surface
        result = asyncio.run(self.engine.generate_code(action))
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.logger.info("Test completed successfully")

    def test_generate_code_sync_wrapper(self):
        self.logger.info("Starting test_generate_code_sync_wrapper")

        current_program = Program(
            id="test-2", code="def add(a, b):\n    return a+b", language="python"
        )

        action = EvolutionAction(
            goal="Add a short docstring",
            instructions=["Add a short docstring"],
            current_program=current_program,
            mode=EvolutionMode.FULL_REWRITE,
        )

        self.logger.info("Executing code generation test with different goal")
        result = asyncio.run(self.engine.generate_code(action))
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.logger.info("Test completed successfully")


if __name__ == "__main__":
    unittest.main()
