import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import ray
from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.actor.actor import ActionResult
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.database.database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Safely format metrics, handling both numeric and string values"""
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:
    """Safely format improvement metrics"""
    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


class Orchestrator:
    def __init__(
        self,
        config,
        initial_program: Program,  # Changed to accept Program object directly
        database,  # Ray actor handle
        evolution_actor: EvolutionActor,
        output_dir: str,
        target_score: "Optional[float]" = None,
        max_iterations: int = 100,
        language: str = "python",
    ):
        self.config = config
        self.initial_program = initial_program  # Store the Program object directly
        self.database = database  # This is now a Ray actor handle
        self.target_score = target_score
        self.evolution_actor = evolution_actor
        self.max_iterations = max_iterations
        self.language = language
        self.output_dir = output_dir

        # Extract file extension from the language or use default
        if language == "python":
            self.file_extension = ".py"
        elif language == "javascript":
            self.file_extension = ".js"
        elif language == "rust":
            self.file_extension = ".rs"
        elif language == "r":
            self.file_extension = ".r"
        else:
            self.file_extension = ".py"  # default

    async def run(self):
        """Run the orchestration process"""
        logger.info("Starting orchestration process")

        # Add initial program to database if it doesn't exist
        ray.get(self.database.add.remote(self.initial_program))

        iteration = 0
        while iteration < self.max_iterations:
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            result: ActionResult = await self.evolution_actor.act()  # This returns a Result object

            if result is None:
                logger.warning("No result from evolution actor, continuing...")
                iteration += 1
                continue

            # Check if there was an error
            if hasattr(result, "error") and result.error:
                logger.warning(f"Iteration {iteration + 1} failed: {result.error}")
                iteration += 1
                continue

            # Check if we have a child program
            if hasattr(result, "child_program_dict") and result.child_program_dict:
                # Create Program object from dictionary (similar to controller logic)
                child_program = Program.from_dict(result.child_program_dict)

                # Add the child program to database using Ray actor
                program_id = ray.get(self.database.add.remote(child_program, iteration=iteration))

                logger.info(f"Iteration {iteration + 1}: Program {program_id} completed")

                if result.artifacts:
                    ray.get(self.database.store_artifacts.remote(program_id, result.artifacts))

                if result.prompt:
                    ray.get(
                        self.database.log_prompt.remote(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )
                    )

            iteration += 1

            # Save the best program at regular intervals
            if iteration % 10 == 0:
                self._save_best_program()

        logger.info("Orchestration process completed")

        # Get and save the final best program
        best_program_result = ray.get(self.database.get_best_program.remote())
        best_program = best_program_result if isinstance(best_program_result, Program) else None

        if best_program:
            logger.info(
                f"Final best program: {best_program.id} with metrics: {_format_metrics(best_program.metrics)}"
            )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return None

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
