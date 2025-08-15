import importlib.util
import unittest
from pathlib import Path

LLM_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "opencontext" / "environment" / "llm_utils.py"
)


def load_llm_utils_module():
    spec = importlib.util.spec_from_file_location("llm_utils", str(LLM_UTILS_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLLMDiffValidatorLocal(unittest.TestCase):
    def test_validator_local(self):
        llm_utils = load_llm_utils_module()

        def sample_current_program():
            return (
                "for i in range(m):\n"
                "\tfor j in range(p):\n"
                "\t\tfor k in range(n):\n"
                "\t\t\tC[i, j] += A[i, k] * B[k, j]\n"
            )

        resp = '[{"search":"for i in range(m):\\n\\tfor j in range(p):\\n\\t\\tfor k in range(n):\\n\\t\\t\\tC[i, j] += A[i, k] * B[k, j]","replace":"for i in range(m):\\n\\tfor k in range(n):\\n\\t\\tfor j in range(p):\\n\\t\\t\\tC[i, j] += A[i, k] * B[k, j]","rationale":"Reorders loops for better locality","order":1}]'
        data = llm_utils.validate_diff_json(resp, sample_current_program())
        self.assertIsInstance(data, list)
        self.assertEqual(data[0].get("order"), 1)


if __name__ == "__main__":
    unittest.main()
