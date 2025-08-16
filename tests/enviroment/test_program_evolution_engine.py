import asyncio
import os
import unittest

from opencontext.common.actions import EvolutionAction
from opencontext.common.program import Program
from opencontext.environment.program_evolution import ProgramEvolutionEngine
from opencontext.llm.llm_openai import OpenAILLM


class TestProgramEvolutionEngineIntegration(unittest.TestCase):
    # Use default OpenAILLM settings (defaults point to localhost-compatible endpoint)
    # Environment variables may override api_base, api_key and model when available.

    def setUp(self):

        self.llm = OpenAILLM()
        self.engine = ProgramEvolutionEngine(self.llm)

    def test_generate_code_async_full_rewrite(self):
        current_program = Program(
            id="test-1", code="def add(a, b):\n    return a+b", language="python"
        )

        action = EvolutionAction(
            task="Add a docstring to the function",
            current_program=current_program,
            mode="full_rewrite",
        )

        # Run the async generation; if it fails the exception will surface
        result = asyncio.get_event_loop().run_until_complete(
            self.engine.generate_code_async(action)
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_generate_code_sync_wrapper(self):
        current_program = Program(
            id="test-2", code="def add(a, b):\n    return a+b", language="python"
        )

        action = EvolutionAction(
            task="Add a short docstring",
            current_program=current_program,
            mode="full_rewrite",
        )
        result = self.engine.generate_code(action)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
