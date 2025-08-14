# we will use ray job client to submit the evaluation job
import os
import re
import tempfile
import time
import logging
import ast

from ray.job_submission import JobSubmissionClient, JobStatus

from opencontext.critic.critic import EvaluationResult, Critic

logger = logging.getLogger(__name__)

# (full code from openevolve/critic/exe_critic.py, with openevolve replaced by opencontext)
