"""
Ray-native evaluation system for OpenEvolve
"""

import importlib.util
import logging
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.utils.ray_task_pool import RayTaskPool
from openevolve.prompt.sampler import PromptSampler

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Ray-native evaluator for programs
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database

        # Ray task pool for parallel evaluation
        self.task_pool = RayTaskPool(max_concurrency=config.parallel_evaluations)

        # Pending artifacts storage
        self._pending_artifacts: Dict[str, Dict[str, any]] = {}

        logger.info(f"Initialized Ray-native evaluator with {evaluation_file}")

    def evaluate_program(self, program_code: str, program_id: str = "") -> Dict[str, float]:
        """Evaluate a single program using Ray"""
        self.task_pool.submit(self._ray_evaluate, program_code, self.evaluation_file)
        results = self.task_pool.get_results()
        return results[0] if results else {"error": 0.0}

    def evaluate_multiple(self, programs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Evaluate multiple programs in parallel using Ray"""
        # Submit all programs to Ray
        for program_code, program_id in programs:
            self.task_pool.submit(self._ray_evaluate, program_code, self.evaluation_file)

        # Get all results
        return self.task_pool.get_results()

    @staticmethod
    def _ray_evaluate(program_code: str, evaluation_file: str) -> Dict[str, float]:
        """Ray worker function - pure synchronous evaluation"""
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(program_code)
                temp_path = f.name

            try:
                # Load evaluation module
                spec = importlib.util.spec_from_file_location("eval_module", evaluation_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Run evaluation
                result = module.evaluate(temp_path)

                # Extract metrics
                if isinstance(result, dict):
                    return result
                elif hasattr(result, "metrics"):
                    return result.metrics
                else:
                    return {"error": 0.0}

            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Ray evaluation failed: {e}")
            return {"error": 0.0}

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, any]]:
        """Get and clear pending artifacts for a program"""
        return self._pending_artifacts.pop(program_id, None)
