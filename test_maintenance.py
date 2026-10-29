"""Checks leakage boundaries, source-order isolation and classification arithmetic."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from main import FAILURE_FLAGS, FEATURES, NUMERIC, TARGET, binary_metrics, evaluate, load_data, split_source_order


class MaintenanceChecks(unittest.TestCase):
    def test_source_labels_and_integrity(self):
        data, audit = load_data()
        self.assertEqual(len(data), 10000)
        self.assertEqual(int(data[TARGET].sum()), 339)
        self.assertEqual(audit["flag_union_disagreements"], 27)
        self.assertTrue(audit["source_is_synthetic"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.csv"
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(path)

    def test_source_order_split(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        self.assertEqual((len(train), len(test)), (8000, 2000))
        self.assertEqual((train["UDI"].iloc[-1], test["UDI"].iloc[0]), (8000, 8001))
        self.assertFalse(set(train["UDI"]) & set(test["UDI"]))
        self.assertEqual((int(train[TARGET].sum()), int(test[TARGET].sum())), (300, 39))

    def test_hand_calculated_positive_class_metrics(self):
        result = binary_metrics([0, 0, 0, 1, 1], [0, 1, 0, 1, 0])
        self.assertEqual((result["true_negative"], result["false_positive"], result["false_negative"], result["true_positive"]), (2, 1, 1, 1))
        self.assertEqual(result["accuracy"], 0.6)
        self.assertEqual(result["failure_precision"], 0.5)
        self.assertEqual(result["failure_recall"], 0.5)
        self.assertEqual(result["failure_f1"], 0.5)
        baseline = binary_metrics([0, 0, 0, 1, 1], [0, 0, 0, 0, 0])
        self.assertEqual(baseline["failure_recall"], 0)
        self.assertEqual(baseline["failure_precision"], 0)

    def test_ids_and_failure_modes_cannot_change_predictions(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        first, before, _ = evaluate(train, test)
        for frame in [train, test]:
            frame[FAILURE_FLAGS] = 1 - frame[FAILURE_FLAGS]
            frame["UDI"] += 50000
            frame["Product ID"] = "IGNORED"
        second, after, _ = evaluate(train, test)
        np.testing.assert_array_equal(before["shallow_tree"], after["shallow_tree"])
        self.assertFalse(set(FEATURES) & set(FAILURE_FLAGS + ["UDI", "Product ID", TARGET]))
        self.assertEqual(first.named_steps["tree"].get_depth(), second.named_steps["tree"].get_depth())

    def test_holdout_labels_do_not_fit_preprocessing_or_model(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        first, before, _ = evaluate(train, test)
        changed = test.copy()
        changed[TARGET] = 1 - changed[TARGET]
        second, after, _ = evaluate(train, changed)
        np.testing.assert_array_equal(before["shallow_tree"], after["shallow_tree"])
        np.testing.assert_allclose(before["weighted_tree_score"], after["weighted_tree_score"])
        medians = first.named_steps["preprocessing"].named_transformers_["numeric"].statistics_
        np.testing.assert_allclose(medians, train[NUMERIC].median())
        self.assertLessEqual(second.named_steps["tree"].get_depth(), 3)

    def test_confusion_counts_and_reproducibility(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        _, predictions, scores = evaluate(train, test)
        _, again, _ = evaluate(train, test)
        np.testing.assert_array_equal(predictions["shallow_tree"], again["shallow_tree"])
        for row in scores.itertuples(index=False):
            self.assertEqual(row.true_negative + row.false_positive + row.false_negative + row.true_positive, len(test))
            self.assertEqual(row.false_negative + row.true_positive, 39)
        self.assertTrue(predictions["weighted_tree_score"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
