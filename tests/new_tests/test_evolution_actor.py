import unittest
import asyncio
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.database import ProgramDatabase, Program
from openevolve.prompt.sampler import PromptSampler
from openevolve.critic.llm_critic import LLMCritic
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.llm.llm_ensemble import EnsembleLLM
from openevolve.config import DatabaseConfig, PromptConfig
from openevolve.llm.config import LLMConfig


class TestEvolutionActor(unittest.TestCase):
    def setUp(self):
        db_config = DatabaseConfig()
        prompt_config = PromptConfig()
        llm_config = LLMConfig()
        self.database = ProgramDatabase(db_config)
        self.prompt_sampler = PromptSampler(prompt_config)
        self.llm_actor_client = EnsembleLLM([llm_config])
        self.llm_critic = LLMCritic(self.llm_actor_client, self.prompt_sampler)
        self.exe_critic = PythonExecutionCritic()
        self.actor = EvolutionActor(
            database=self.database,
            prompt_sampler=self.prompt_sampler,
            llm_actor_client=self.llm_actor_client,
            llm_critic=self.llm_critic,
            exe_critic=self.exe_critic,
            language="python",
            iteration=1,
            diff_based_evolution=False,
            max_code_length=1000,
            use_llm_critic=True,
            llm_feedback_weight=0.1,
            artifacts_enabled=False,
        )
        # Add a parent program to the database
        parent_program = Program(
            id="parent1",
            code="print('hello')",
            language="python",
            parent_id=None,
            generation=0,
            metrics={"score": 1.0},
            iteration_found=0,
            metadata={},
        )
        self.database.programs[parent_program.id] = parent_program
        self.database.islands[0].add(parent_program.id)

    def test_evolution_actor_act_returns_result(self):
        async def run_act():
            result = await self.actor.act()
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.child_program)
            from openevolve.database import Program

            self.assertIsInstance(result.child_program, Program)
            self.assertEqual(result.child_program.code, "def foo():\n    return 42")

        asyncio.run(run_act())


if __name__ == "__main__":
    unittest.main()
