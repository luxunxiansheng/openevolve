import unittest
import asyncio
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.database.database import ProgramDatabase, Program
from openevolve.prompt.sampler import PromptSampler
from openevolve.critic.llm_critic import LLMCritic
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.llm.llm_ensemble import EnsembleLLM
from openevolve.database.config import DatabaseConfig
from openevolve.prompt.config import PromptConfig
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

        )
        
        python_file_path = "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py"  # Replace with an actual script path
        
        with open(python_file_path, "r") as file:
            python_code = file.read()

        parent_program = Program(
            id="parent1",
            code=python_code,
            language="python",
            parent_id=None,
            generation=0,
            metrics={},
            iteration_found=0,
            metadata={},
        )
        self.database.add(parent_program)
   

    def test_evolution_actor_act_returns_result(self):
        async def run_act():
            result = await self.actor.act()
            print("Action Result:", result)
            self.assertIsNotNone(result)
           
        asyncio.run(run_act())


if __name__ == "__main__":
    unittest.main()
