import asyncio
import unittest
from openevolve.llm.openai_llm import OpenAILLM
from openevolve.llm.config import LLMConfig

class TestOpenAILLM(unittest.TestCase):
    def setUp(self):
        # Use OpenAILLM with a default LLMConfig object
        self.llm = OpenAILLM(LLMConfig())

    def test_generate(self):
        async def run():
            result = await self.llm.generate("can you program a python app?")
            print("Result:", result)
            self.assertIsInstance(result, str)
        asyncio.run(run())

    def test_generate_with_context(self):
        async def run():
            result = await self.llm.generate_with_context(
                system_message="Test system",
                messages=[{"role": "user", "content": "can you program a python app"}]
            )
            print("Result:", result)
            self.assertIsInstance(result, str)
        asyncio.run(run())

if __name__ == "__main__":
    unittest.main()
