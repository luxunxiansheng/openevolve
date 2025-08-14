import logging
import os
import time
from typing import Optional
import uuid

from networkx import number_of_isolates
import ray

from openevolve.actor.actor import ActionResult
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.critic.llm_critic import LLMCritic
from openevolve.database.database import Program, ProgramDatabase
from openevolve.llm.llm_ensemble import EnsembleLLM
from ..llm.llm_openai import OpenAILLM
from openevolve.prompt.sampler import PromptSampler
from openevolve.prompt.templates import Templates
from openevolve.utils.metrics_utils import safe_numeric_average


logger = logging.getLogger(__name__)


class Orchestrator:

    def __init__(
        self,
        critic_program_path,
        evoved_program_path,
        output_dir: str,
        # Orchestrator config fields
        max_iterations: int = 1000,
        target_score: float = 1.0,
        file_extension: str = ".py",
        language: str = "python",
        diff_based_evolution: bool = True,
        iterations_per_island: int = 15,
        # Database config fields
        db_num_islands: int = 5,
    ):
        self.output_dir = output_dir
        self.evolved_program_path = evoved_program_path
        self.file_extension = file_extension
        self.language = language
        self.diff_based_evolution = diff_based_evolution
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.programs_per_island = max_iterations // (iterations_per_island * db_num_islands)

        # Initialize the program database
        self.database = ray.remote(ProgramDatabase).remote(num_islands=db_num_islands)

        actor_prompt_sampler = PromptSampler(system_template_key=Templates.ACTOR_SYSTEM)
        llm_actor_client = EnsembleLLM([OpenAILLM()])

        critic_prompt_sampler = PromptSampler(system_template_key=Templates.CRITIC_SYSTEM)
        llm_critic_client = EnsembleLLM([OpenAILLM()])
        llm_critic = LLMCritic(llm_critic_client, critic_prompt_sampler)

        exe_critic = PythonExecutionCritic(critic_program_path=critic_program_path)

        # Initialize the EvolutionActor with the database and critics
        self.evolution_actor = EvolutionActor(
            database=self.database,
            actor_prompt_sampler=actor_prompt_sampler,
            llm_actor_client=llm_actor_client,
            llm_critic=llm_critic,
            exe_critic=exe_critic,
        )

    async def run(self):
        """Run the orchestration process"""
        logger.info("Starting orchestration process")

        with open(self.evolved_program_path, "r") as file:
            program_to_be_evolved = file.read()

        if not program_to_be_evolved:
            raise ValueError("No valid program found in the provided file.")

        # Create initial program object
        initial_evolved_program = Program(
            id=str(uuid.uuid4()),
            code=program_to_be_evolved,
            language=self.language,
            parent_id=None,
            generation=0,
            metrics={},
            iteration_found=0,
            metadata={},
        )

        # Add initial program to database if it doesn't exist
        ray.get(self.database.add.remote(initial_evolved_program))

        # Initialize island tracking variables
        start_iteration = 0
        current_island_counter = 0

        current_iteration = 0

        while current_iteration < self.max_iterations:
            logger.info(f"Running iteration {current_iteration + 1}/{self.max_iterations}")
            try:
                logger.info(ray.get(self.database.log_island_status.remote()))

                result: ActionResult = await self.evolution_actor.act(
                    iteration=current_iteration
                )  # This returns a Result object

                logger.debug(f"Result from evolution actor: {result}")

                # Reconstruct program from dict
                child_program = Program(**result.child_program_dict)

                # Add to database
                ray.get(self.database.add.remote(child_program, iteration=current_iteration))

                # Store artifacts
                if result.artifacts:
                    ray.get(
                        self.database.store_artifacts.remote(child_program.id, result.artifacts)
                    )

                # Island management
                if (
                    current_iteration > start_iteration
                    and current_island_counter >= self.programs_per_island
                ):
                    ray.get(self.database.next_island.remote())
                    current_island_counter = 0
                    current_island = ray.get(self.database.get_current_island.remote())
                    logger.info(f"Switched to island {current_island}")

                current_island_counter += 1
                ray.get(self.database.increment_island_generation.remote())

                # Check migration
                should_migrate = ray.get(self.database.should_migrate.remote())
                if should_migrate:
                    logger.info(f"Performing migration at iteration {current_iteration}")
                    ray.get(self.database.migrate_programs.remote())
                    ray.get(self.database.log_island_status.remote())

                # Log progress
                logger.info(
                    f"Iteration {current_iteration}: "
                    f"Program {child_program.id} "
                    f"(parent: {result.parent_id}) "
                    f"completed in {result.iteration_time:.2f}s"
                )

                avg_score = safe_numeric_average(child_program.metrics)
                logger.info(
                    f"Iteration {current_iteration}: "
                    f"Average score for program {child_program.id} is {avg_score:.2f}"
                )

                # Check for new best
                best_program_id = ray.get(self.database.get_best_program_id.remote())
                if best_program_id == child_program.id:
                    logger.info(
                        f"🌟 New best solution found at iteration {current_iteration}: "
                        f"{child_program.id}"
                    )
                    # Save the new best program
                    self._save_best_program(child_program)

                # Check target score
                if self.target_score is not None:
                    if avg_score >= self.target_score:
                        logger.info(
                            f"Target score {self.target_score} reached at iteration {current_iteration}"
                        )
                        break

            except Exception as e:
                logger.error(f"Error processing result from iteration {current_iteration}: {e}")

            current_iteration += 1

        # Save final best program
        logger.info("Evolution completed. Saving final best program...")
        self._save_best_program()

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, get the best program from the database
        if program is None:
            best_program_result = ray.get(self.database.get_best_program.remote())
            program = best_program_result if isinstance(best_program_result, Program) else None

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
