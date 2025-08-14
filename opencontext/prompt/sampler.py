
"""
Prompt sampling for OpenContext
"""

import logging
import random
from typing import Any, Dict, List, Optional, Union

from opencontext.prompt.templates import TemplateManager, Templates
from opencontext.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)

class PromptSampler:
	"""Generates prompts for code evolution with configurable templates and formatting"""

	def __init__(
		self,
		template_dir: Optional[str] = None,
		system_template_key: str = Templates.BASE_SYSTEM,
		num_top_programs: int = 3,
		num_diverse_programs: int = 2,
		use_template_stochasticity: bool = True,
		template_variations: Optional[Dict[str, List[str]]] = None,
		use_meta_prompting: bool = False,
		meta_prompt_weight: float = 0.1,
		include_artifacts: bool = True,
		max_artifact_bytes: int = 20 * 1024,
		include_changes_under_chars: int = 100,
		concise_implementation_max_lines: int = 10,
		comprehensive_implementation_min_lines: int = 50,
		code_length_threshold: Optional[int] = 2048,
	):
		# ...existing code from openevolve/prompt/sampler.py, with openevolve replaced by opencontext...
