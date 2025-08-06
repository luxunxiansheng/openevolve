from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30

    def to_dict(self) -> Dict[str, Optional[Union[str, bool, int, float, List[str], Dict[str, int]]]]:
        """Convert configuration to a dictionary"""
        return {
            "db_path": self.db_path,
            "in_memory": self.in_memory,
            "log_prompts": self.log_prompts,
            "population_size": self.population_size,
            "archive_size": self.archive_size,
            "num_islands": self.num_islands,
            "elite_selection_ratio": self.elite_selection_ratio,
            "exploration_ratio": self.exploration_ratio,
            "exploitation_ratio": self.exploitation_ratio,
            "diversity_metric": self.diversity_metric,
            "feature_dimensions": self.feature_dimensions,
            "feature_bins": self.feature_bins,
            "diversity_reference_size": self.diversity_reference_size,
            "migration_interval": self.migration_interval,
            "migration_rate": self.migration_rate,
            "random_seed": self.random_seed,
            "artifacts_base_path": self.artifacts_base_path,
            "artifact_size_threshold": self.artifact_size_threshold,
            "cleanup_old_artifacts": self.cleanup_old_artifacts,
            "artifact_retention_days": self.artifact_retention_days
        }
