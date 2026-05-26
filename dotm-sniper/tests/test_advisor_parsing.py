#!/usr/bin/env python3
"""
Tests for advisor_script.py LLM response parsing and schema validation.
"""
import unittest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from advisor_script import parse_llm_advisor_response, _validate_advisor_schema


VALID_RESULT = {
    "p_estimate": 0.65,
    "confidence": 0.80,
    "factors": ["Factor A", "Factor B"],
    "verdict": "CONFIRM"
}


class TestParseCleanJSON(unittest.TestCase):
    def test_clean_json(self):
        raw = json.dumps(VALID_RESULT)
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "CONFIRM")
        self.assertAlmostEqual(result["p_estimate"], 0.65)

    def test_json_with_whitespace(self):
        raw = "  \n  " + json.dumps(VALID_RESULT) + "  \n  "
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "CONFIRM")


class TestParseFencedCodeBlock(unittest.TestCase):
    def test_json_in_fenced_block(self):
        raw = "```json\n" + json.dumps(VALID_RESULT) + "\n```"
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "CONFIRM")

    def test_json_in_fenced_block_with_preamble(self):
        raw = "Here is my analysis:\n\n```json\n" + json.dumps(VALID_RESULT) + "\n```\n\nDone."
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "CONFIRM")


class TestParsePreamble(unittest.TestCase):
    def test_json_after_preamble_text(self):
        raw = "I have analyzed the market. Here is my response:\n" + json.dumps(VALID_RESULT)
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "CONFIRM")

    def test_json_with_leading_explanation(self):
        raw = "The market looks favorable.\nBased on my analysis:\n" + json.dumps(VALID_RESULT) + "\nThat is my conclusion."
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertAlmostEqual(result["p_estimate"], 0.65)


class TestParseBraceBalancing(unittest.TestCase):
    def test_nested_json(self):
        obj = {
            "p_estimate": 0.5,
            "confidence": 0.7,
            "factors": ["A"],
            "verdict": "WARNING",
            "extra": {"nested": {"deep": True}}
        }
        raw = "Analysis:\n" + json.dumps(obj) + "\nEnd."
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "WARNING")
        self.assertEqual(result["extra"]["nested"]["deep"], True)

    def test_json_with_trailing_text_after_object(self):
        raw = json.dumps(VALID_RESULT) + "\nSome trailing text here."
        result, err = parse_llm_advisor_response(raw)
        self.assertIsNone(err)
        self.assertEqual(result["verdict"], "CONFIRM")


class TestParseFailures(unittest.TestCase):
    def test_empty_input(self):
        result, err = parse_llm_advisor_response("")
        self.assertIsNone(result)
        self.assertIn("empty", err)

    def test_none_input(self):
        result, err = parse_llm_advisor_response(None)
        self.assertIsNone(result)
        self.assertIn("empty", err)

    def test_no_json_at_all(self):
        result, err = parse_llm_advisor_response("Just some text without JSON")
        self.assertIsNone(result)
        self.assertIn("{", err)

    def test_malformed_json(self):
        result, err = parse_llm_advisor_response('{"broken": json}')
        self.assertIsNone(result)


class TestSchemaValidation(unittest.TestCase):
    def test_valid_schema(self):
        err = _validate_advisor_schema(VALID_RESULT)
        self.assertIsNone(err)

    def test_missing_p_estimate(self):
        obj = dict(VALID_RESULT)
        del obj["p_estimate"]
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("p_estimate", err)

    def test_missing_confidence(self):
        obj = dict(VALID_RESULT)
        del obj["confidence"]
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("confidence", err)

    def test_missing_factors(self):
        obj = dict(VALID_RESULT)
        del obj["factors"]
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("factors", err)

    def test_missing_verdict(self):
        obj = dict(VALID_RESULT)
        del obj["verdict"]
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("verdict", err)

    def test_p_estimate_out_of_range(self):
        obj = dict(VALID_RESULT, p_estimate=1.5)
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("p_estimate", err)

    def test_p_estimate_negative(self):
        obj = dict(VALID_RESULT, p_estimate=-0.1)
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("p_estimate", err)

    def test_invalid_verdict(self):
        obj = dict(VALID_RESULT, verdict="MAYBE")
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("verdict", err)

    def test_factors_not_list(self):
        obj = dict(VALID_RESULT, factors="not a list")
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("factors", err)

    def test_factors_contain_non_string(self):
        obj = dict(VALID_RESULT, factors=[1, 2])
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(result)
        self.assertIn("factors", err)

    def test_all_valid_verdicts(self):
        for v in ["CONFIRM", "DIVERGE", "WARNING", "UNKNOWN"]:
            obj = dict(VALID_RESULT, verdict=v)
            result, err = parse_llm_advisor_response(json.dumps(obj))
            self.assertIsNone(err, f"verdict={v} should be valid")
            self.assertEqual(result["verdict"], v)

    def test_non_dict_input(self):
        result, err = parse_llm_advisor_response('[1, 2, 3]')
        self.assertIsNone(result)

    def test_boundary_p_estimate_zero(self):
        obj = dict(VALID_RESULT, p_estimate=0.0)
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(err)

    def test_boundary_p_estimate_one(self):
        obj = dict(VALID_RESULT, p_estimate=1.0)
        result, err = parse_llm_advisor_response(json.dumps(obj))
        self.assertIsNone(err)


if __name__ == '__main__':
    unittest.main(verbosity=2)
