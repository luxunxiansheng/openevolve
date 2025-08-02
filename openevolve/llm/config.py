from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM models"""

  
    # Model identification
    name: str = "Qwen3-14B-AWQ"

    # API/model configuration
    api_base: str = "http://localhost:8010/v1"
    api_key: Optional[str] = "none"
   

    # Weight for model in ensemble (will be normalized)
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 20480

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # Reproducibility
    random_seed: Optional[int] = 0

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.weight <= 0:
            raise ValueError(f"Model weight must be positive, got {self.weight}")


@dataclass
class EnsembleLLMConfig:
    """Configuration for an ensemble of LLM models"""

    # List of models in the ensemble
    ensemble_models: List[LLMConfig] = field(default_factory=lambda: [
        LLMConfig()
    ])
    
    # Ensemble behavior configuration
    generate_all_with_context: bool = False  # If True, use all models and combine results
    # You can add more ensemble-level defaults here if needed
    
    # Shared configuration that applies to all models if not individually specified
    shared_api_base: Optional[str] = None
    shared_api_key: Optional[str] = None
    shared_temperature: Optional[float] = None
    shared_top_p: Optional[float] = None
    shared_max_tokens: Optional[int] = None
    shared_timeout: Optional[int] = None
    shared_retries: Optional[int] = None
    shared_retry_delay: Optional[int] = None
    shared_random_seed: Optional[int] = None

    def __post_init__(self):
        """Post-initialization to validate and set up ensemble"""
        if not self.ensemble_models:
            raise ValueError("Ensemble must have at least one model")
            
        # Apply shared configuration to models that don't have specific values
        shared_config = {
            "api_base": self.shared_api_base,
            "api_key": self.shared_api_key, 
            "temperature": self.shared_temperature,
            "top_p": self.shared_top_p,
            "max_tokens": self.shared_max_tokens,
            "timeout": self.shared_timeout,
            "retries": self.shared_retries,
            "retry_delay": self.shared_retry_delay,
            "random_seed": self.shared_random_seed,
        }
        
        for model in self.ensemble_models:
            for key, value in shared_config.items():
                if value is not None and getattr(model, key, None) is None:
                    setattr(model, key, value)
        
        # Validate weights
        total_weight = sum(model.weight for model in self.ensemble_models)
        if total_weight <= 0:
            raise ValueError("Total ensemble weight must be positive")
            
    def add_model(self, model_config: LLMConfig) -> None:
        """Add a model to the ensemble"""
        self.ensemble_models.append(model_config)
        
    def get_total_weight(self) -> float:
        """Get the total weight of all models in the ensemble"""
        return sum(model.weight for model in self.ensemble_models)
        
    def get_normalized_weights(self) -> List[float]:
        """Get normalized weights for all models"""
        total = self.get_total_weight()
        return [model.weight / total for model in self.ensemble_models]

    @property
    def weights(self) -> List[float]:
        """Property to get weights (for compatibility with ensemble.py)"""
        return [model.weight for model in self.ensemble_models]

