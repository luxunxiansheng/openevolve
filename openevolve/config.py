"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .database.config import DatabaseConfig

from .prompt.config import PromptConfig

from .llm.config import LLMConfig
from .evaluator.config import EvaluatorConfig

@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 100000

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                # Note: diversity_metric fixed to "edit_distance"
                # "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator.memory_limit_mb,
                # "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "max_code_length": self.max_code_length,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
