"""Response-contract checks only; these fixtures are not Jev measurements."""

import unittest

from collect_jev import MODEL, parse_response


def response():
    return {"model": MODEL, "answers": {"topic": {
        "type": "choice", "choice": "world",
        "probabilities": {"world": 1.0, "sports": 0.0, "business": 0.0, "sci_tech": 0.0},
    }}}


class JevContractTests(unittest.TestCase):
    def test_true_zero_probabilities_are_preserved(self):
        self.assertEqual(parse_response(response()), [1.0, 0.0, 0.0, 0.0])

    def test_boolean_probability_rejected(self):
        body = response()
        body["answers"]["topic"]["probabilities"]["world"] = True
        with self.assertRaises(ValueError):
            parse_response(body)

    def test_version_drift_rejected(self):
        body = response()
        body["model"] = "jev-latest"
        with self.assertRaises(ValueError):
            parse_response(body)

    def test_non_object_rejected(self):
        for body in ([], None, {"model": MODEL, "answers": []}):
            with self.assertRaises(ValueError):
                parse_response(body)

    def test_choice_disagreement_rejected(self):
        body = response()
        body["answers"]["topic"]["choice"] = "sports"
        with self.assertRaises(ValueError):
            parse_response(body)


if __name__ == "__main__":
    unittest.main()
