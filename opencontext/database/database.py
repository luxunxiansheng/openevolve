"""
Program database for OpenContext
"""

import base64
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields

# FileLock removed - no longer needed with threaded parallel processing
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from opencontext.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)

# ...rest of the code from openevolve/database/database.py, with openevolve replaced by opencontext...
