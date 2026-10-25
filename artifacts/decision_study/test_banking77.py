"""Check shortlist class mapping, missing-gold accounting, and split isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from banking77 import freeze, rerank_distribution, top_candidates, write_json
from metrics import quality


class RoutingTests(unittest.TestCase):
    def test_scatter_uses_each_requests_class_order(self):
        ids = np.array([[1, 3], [0, 2]])
        p = rerank_distribution([[2, 0], [0, 2]], ids, 4)
        self.assertEqual(p.argmax(1).tolist(), [1, 2])
        np.testing.assert_allclose(p.sum(1), 1)
        self.assertEqual(p[0, 0], 0)
        self.assertEqual(p[1, 3], 0)

    def test_shortlist_miss_stays_a_full_system_error(self):
        p = rerank_distribution([[2, 0]], np.array([[0, 1]]), 3)
        result = quality(p, [2])
        self.assertEqual(result["accuracy"], 0)
        self.assertAlmostEqual(result["nll"], -np.log(1e-12))

    def test_prior_is_restricted_without_changing_global_ids(self):
        prior = np.array([[.1, .6, .3]])
        p = rerank_distribution([[0, 0]], np.array([[0, 2]]), 3, prior=prior)
        np.testing.assert_allclose(p, [[.25, 0, .75]])

    def test_topk_has_deterministic_ties_and_canonical_order(self):
        ids = top_candidates(np.array([[.2, .6, .2], [.5, .4, .1]]), k=2)
        self.assertEqual(ids.tolist(), [[0, 1], [0, 1]])

    def test_selection_file_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "selection.json"
            write_json(path, {"selected": "first"})
            with self.assertRaises(FileExistsError):
                write_json(path, {"selected": "second"})

    def test_official_data_splits_do_not_overlap(self):
        data = Path(__file__).parent / ".cache/banking77"
        if not data.exists():
            data = Path(__file__).parent / "data/banking77"
        if not data.exists():
            self.skipTest("Verified official source not downloaded")
        labels, splits, _ = freeze(data)
        self.assertEqual(len(labels), 77)
        fingerprints = {name: {r["text_sha256"] for r in rows}
                        for name, rows in splits.items()}
        self.assertFalse(fingerprints["fit"] & fingerprints["select"])
        self.assertFalse(fingerprints["fit"] & fingerprints["test"])
        self.assertFalse(fingerprints["select"] & fingerprints["test"])
        self.assertEqual(np.bincount([r["label"] for r in splits["select"]]).tolist(), [10] * 77)


if __name__ == "__main__":
    unittest.main()
