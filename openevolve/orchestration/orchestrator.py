import logging
import os
from typing import Any, Dict, Optional


from openevolve.actor.evolution_actor import EvolutionActor
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
    def __init__(self, 
                 config,
                 initial_program_path: str,
                 database: ProgramDatabase,
                 evolution_actor: EvolutionActor,
                 output_dir: str = "output",
         
                 target_score: 'Optional[float]' = None,
                 max_iterations: int = 100,
                 language: str = "python",
                 ):
        self.config = config
        self.initial_program_path = initial_program_path
        self.database = database
        self.target_score = target_score
        self.evolution_actor = evolution_actor
        self.max_iterations = max_iterations
        self.language = language    
        self.output_dir = output_dir

    def run(self):
        """Run the orchestration process"""
        logger.info("Starting orchestration process")


        initial_program = self._create_initial_program()
        self.database.add_program(initial_program)
      
  
        while iteration < self.max_iterations:
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            result = self.evolution_actor.act()
            if result is None or result.child_program is None:
                logger.warning("No valid child program generated, stopping evolution.")
                break

            child_program = result.child_program
            metrics = child_program.metrics
            improvement = result.improvement

            logger.info(f"Child Program ID: {child_program.id}")
            logger.info(f"Metrics: {_format_metrics(metrics)}")
            logger.info(f"Improvement: {_format_improvement(improvement)}")

            iteration += 1

        logger.info("Orchestration process completed.")
        
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if best_program and "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs "
                        f"{best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
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
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

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



    

    

        

