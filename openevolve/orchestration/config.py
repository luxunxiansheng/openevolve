from dataclasses import dataclass
from typing import Optional


@dataclass
class OrchestratorConfig:
    max_iterations: int = 100
    language: str = "python"
    iterations_per_island:int = 10
    diff_based_evolution: bool = False
    target_score: Optional[float] = 2.5
    file_extension: str = ".py"
