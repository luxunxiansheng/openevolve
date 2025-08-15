import unittest

from opencontext.environment import llm_utils


def sample_current_program():
    return (
        "for i in range(m):\n"
        "\tfor j in range(p):\n"
        "\t\tfor k in range(n):\n"
        "\t\t\tC[i, j] += A[i, k] * B[k, j]\n"
    )


class TestLLMDiffValidator(unittest.TestCase):
    def test_valid_response(self):
        resp = '[{"search":"for i in range(m):\\n\\tfor j in range(p):\\n\\t\\tfor k in range(n):\\n\\t\\t\\tC[i, j] += A[i, k] * B[k, j]","replace":"for i in range(m):\\n\\tfor k in range(n):\\n\\t\\tfor j in range(p):\\n\\t\\t\\tC[i, j] += A[i, k] * B[k, j]","rationale":"Reorders loops for better locality","order":1}]'
        data = llm_utils.validate_diff_json(resp, sample_current_program())
        self.assertIsInstance(data, list)
        self.assertEqual(data[0].get("order"), 1)

    def test_invalid_json(self):
        with self.assertRaises(llm_utils.ValidationError):
            llm_utils.validate_diff_json("not json", sample_current_program())

    def test_search_not_found(self):
        bad = '[{"search":"nonexistent","replace":"x","rationale":"r","order":1}]'
        with self.assertRaises(llm_utils.ValidationError):
            llm_utils.validate_diff_json(bad, sample_current_program())


if __name__ == "__main__":
    unittest.main()
