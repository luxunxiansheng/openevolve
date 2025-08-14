"""
LLM-based evaluator for program assessment

Simplified implementation without prompt_sampler dependency.
"""

import json
import logging
import re
from typing import Dict

from opencontext.llm.llm_interface import LLMInterface
from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# ...rest of the code from openevolve/environment/evaluators/llm_evaluator.py, with openevolve replaced by opencontext...
