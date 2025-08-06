import unittest
import asyncio
import ray

from openevolve.actor.evolution_actor import EvolutionActor

from openevolve.critic.exe_critic import PythonExecutionCritic

from openevolve.critic.llm_critic import LLMCritic

from openevolve.llm.llm_ensemble import EnsembleLLM

from openevolve.orchestration.orchestrator import Orchestrator
from openevolve.prompt.sampler import PromptSampler

from openevolve.database.database import Program, ProgramDatabase

from openevolve.llm.config import LLMConfig

from openevolve.prompt.config import PromptConfig

# Import the Orchestrator and its config
from openevolve.database.config import DatabaseConfig

python_file_path = "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py"  # Replace with an actual script path
output_path = "/workspaces/openevolve/tests/outputs"      

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        db_config = DatabaseConfig()
        prompt_config = PromptConfig()
        llm_config = LLMConfig()
        self.database = ray.remote(ProgramDatabase).remote(db_config)
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
        
      
        with open(python_file_path, "r") as file:
            python_code = file.read()

        init_program = Program(
            id="parent1",
            code=python_code,
            language="python",
            parent_id=None,
            generation=0,
            metrics={},
            iteration_found=0,
            metadata={},
        )

        self.orchestrator = Orchestrator(
            initial_program=init_program,
            database=self.database,
            evolution_actor=self.actor,
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
