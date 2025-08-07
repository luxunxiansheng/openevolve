import logging
import os
import time
from typing import Optional

import ray
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.actor.actor import ActionResult
from openevolve.database.database import Program
from openevolve.orchestration.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        initial_program: Program,
        database,
        evolution_actor: EvolutionActor,
        output_dir: str,
        config: OrchestratorConfig = OrchestratorConfig(),
    ):
        self.initial_program = initial_program
        self.database = database
        self.evolution_actor = evolution_actor
        self.output_dir = output_dir
        self.config = config
        self.target_score = config.target_score
        self.max_iterations = config.max_iterations
        self.language = config.language
        self.programs_per_island = config.programs_per_island
        self.diff_based_evolution = config.diff_based_evolution

        # Extract file extension from the language or use default
        if self.language == "python":
            self.file_extension = ".py"
        elif self.language == "javascript":
            self.file_extension = ".js"
        elif self.language == "rust":
            self.file_extension = ".rs"
        elif self.language == "r":
            self.file_extension = ".r"
        else:
            self.file_extension = ".py"  # default

    async def run(self):
        """Run the orchestration process"""
        logger.info("Starting orchestration process")

        # Add initial program to database if it doesn't exist
        ray.get(self.database.add.remote(self.initial_program))

        # Initialize island tracking variables
        start_iteration = 0
        current_island_counter = 0

        completed_iteration = 0

        while completed_iteration < self.max_iterations:
            result: ActionResult = await self.evolution_actor.act()  # This returns a Result object
            try:
                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to database
                    ray.get(self.database.add.remote(child_program, iteration=completed_iteration))

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
                                    "full_rewrite_user"
                                    if not self.diff_based_evolution
                                    else "diff_user"
                                ),
                                program_id=child_program.id,
                                prompt=result.prompt,
                                responses=[result.llm_response] if result.llm_response else [],
                            )
                        )

                    # Island management
                    if (
                        completed_iteration > start_iteration
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
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        ray.get(self.database.migrate_programs.remote())
                        ray.get(self.database.log_island_status.remote())

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

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
                            from openevolve.utils.metrics_utils import safe_numeric_average

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
                            f"🌟 New best solution found at iteration {completed_iteration}: "
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
                                    f"Target score {self.target_score} reached at iteration {completed_iteration}"
                                )
                                break

            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iteration += 1

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
