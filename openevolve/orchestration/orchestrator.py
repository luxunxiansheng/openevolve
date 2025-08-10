import logging
import os
import time
from typing import Optional
import uuid

import ray

from openevolve.actor.actor import ActionResult
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.critic.llm_critic import LLMCritic
from openevolve.database.database import Program, ProgramDatabase
from openevolve.llm.llm_ensemble import EnsembleLLM
from openevolve.prompt.sampler import PromptSampler
from openevolve.prompt.templates import TemplateKey
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
        iterations_per_island: int = 10,
        # Database config fields
        db_num_islands: int = 5,
        db_other_kwargs: dict = {},
        # PromptSampler config fields
        prompt_kwargs: dict = {},
        # LLM config fields (list of dicts for ensemble)
        llm_model_cfgs: list = None,
        llm_weights: list = None,
    ):
        self.evolved_program_path = evoved_program_path
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.file_extension = file_extension
        self.language = language
        self.diff_based_evolution = diff_based_evolution
        self.programs_per_island = max(
            1,
            max_iterations // (db_num_islands * iterations_per_island),
        )

        # Database
        db_kwargs = dict(num_islands=db_num_islands, **db_other_kwargs)
        self.database = ray.remote(ProgramDatabase).remote(**db_kwargs)

        # LLM ensemble
        llm_model_cfgs = llm_model_cfgs or [{}]
        llm_client = EnsembleLLM(
            [
                # Each dict in llm_model_cfgs is passed as kwargs
                type("Dummy", (), cfg)() if isinstance(cfg, dict) else cfg
                for cfg in llm_model_cfgs
            ],
            weights=llm_weights,
        )

        # PromptSampler
        prompt_sampler = PromptSampler(**prompt_kwargs)
        llm_critic = LLMCritic(llm_client, prompt_sampler)
        exe_critic = PythonExecutionCritic(critic_program_path=critic_program_path)

        self.evolution_actor = EvolutionActor(
            database=self.database,
            actor_prompt_sampler=prompt_sampler,
            llm_actor_client=llm_client,
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
                if result.error:
                    logger.warning(f"Iteration {current_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to database
                    ray.get(self.database.add.remote(child_program, iteration=current_iteration))

                    # Store artifacts
                    if result.artifacts:
                        ray.get(
                            self.database.store_artifacts.remote(child_program.id, result.artifacts)
                        )

                    # Log prompts
                    if result.prompt:
                        ray.get(
                            self.database.log_prompt.remote(
                                template_key=(
                                    TemplateKey.FULL_REWRITE_USER.value
                                    if not self.diff_based_evolution
                                    else TemplateKey.DIFF_USER.value
                                ),
                                program_id=child_program.id,
                                prompt=result.prompt,
                                responses=[result.llm_response] if result.llm_response else [],
                            )
                        )

                    # Island management
                    if (
                        current_iteration > start_iteration
                        and current_island_counter >= self.programs_per_island
                    ):
                        ray.get(self.database.next_island.remote())
                        current_island_counter = 0
                        current_island = ray.get(self.database.get_current_island.remote())
                        logger.debug(f"Switched to island {current_island}")

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

                    avg_score = 0.0
                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):

                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

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
                    if self.target_score is not None and child_program.metrics:
                        numeric_metrics = [
                            v for v in child_program.metrics.values() if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
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
