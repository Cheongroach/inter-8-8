"""Numerical checks for the experiment's independent scoring implementation."""

import unittest

import numpy as np

from metrics import apply_affine, fit_affine, fit_temperature, paired_accuracy, probabilities, quality


class QualityTests(unittest.TestCase):
    def test_uniform_four_class_distribution(self):
        result = quality(np.full((4, 4), 0.25), [0, 1, 2, 3])
        self.assertAlmostEqual(result["nll"], np.log(4))
        self.assertAlmostEqual(result["brier_multiclass_sum"], 0.75)
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["ece_15"], 0.0)

    def test_confidence_one_is_in_last_bin(self):
        result = quality([[1.0, 0.0], [1.0, 0.0]], [0, 1])
        self.assertEqual(result["ece_15"], 0.5)

    def test_temperature_does_not_change_argmax(self):
        z = np.array([[2, -1, 0], [-3, 2, 1]])
        self.assertTrue(np.array_equal(probabilities(z, 0.5).argmax(1),
                                       probabilities(z, 8).argmax(1)))

    def test_identity_affine_map_and_common_logit_shift(self):
        z = np.array([[2, -1, 0], [-3, 2, 1]], dtype=float)
        np.testing.assert_allclose(apply_affine(z, np.eye(3), np.zeros(3)), probabilities(z))
        np.testing.assert_allclose(probabilities(z + 1000), probabilities(z))

    def test_invalid_probability_row_rejected(self):
        with self.assertRaises(ValueError):
            quality([[0.2, 0.2]], [0])
        with self.assertRaises(ValueError):
            quality([[0.5, 0.5]], [2])

    def test_paired_bootstrap_uses_cases_not_independent_arms(self):
        p = [[0.8, 0.2], [0.2, 0.8], [0.8, 0.2]]
        result = paired_accuracy(p, p, [0, 1, 1])
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertEqual(result["wins"], 0)
        self.assertEqual(result["losses"], 0)

    def test_probability_fit_uses_supplied_labels(self):
        z = np.tile([10.0, 0.0], (20, 1))
        labels = [0, 1] * 10
        before = quality(probabilities(z), labels)["nll"]
        t = fit_temperature(z, labels)
        after = quality(probabilities(z, t), labels)["nll"]
        self.assertLess(after, before)
        w, b = fit_affine(z, labels, 0.1)
        self.assertLess(quality(apply_affine(z, w, b), labels)["nll"], before)


    def test_affine_prior_preserves_shipped_temperature(self):
        z = np.array([[3.0, -1.0], [1.0, 2.0]])
        temperature = 1.76
        w, b = fit_affine(z / temperature, [0, 1], 1.0, steps=0)
        np.testing.assert_allclose(
            apply_affine(z / temperature, w, b), probabilities(z, temperature)
        )


if __name__ == "__main__":
    unittest.main()
