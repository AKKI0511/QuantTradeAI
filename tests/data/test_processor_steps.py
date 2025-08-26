import unittest
import pandas as pd
import numpy as np
import os

from quanttradeai.data.processor import DataProcessor


class TestProcessorSteps(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()

    def test_generate_feature_combinations(self):
        self.processor.cross_indicators = [["a", "b"]]
        self.processor.ratio_indicators = [["a", "b"]]
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 4]})
        result = self.processor._generate_feature_combinations(df.copy())
        self.assertIn("cross_a_b", result.columns)
        self.assertIn("ratio_a_b", result.columns)

    def test_scale_features_minmax(self):
        self.processor.scaling_method = "minmax"
        self.processor.scaling_range = [0, 1]
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scaled = self.processor._scale_features(df.copy())
        self.assertAlmostEqual(scaled["a"].min(), 0.0)
        self.assertAlmostEqual(scaled["a"].max(), 1.0)

    def test_select_features(self):
        self.processor.n_features = 2
        df = pd.DataFrame(np.random.rand(3, 4), columns=list("abcd"))
        selected = self.processor._select_features(df)
        self.assertEqual(list(selected.columns), ["a", "b"])

    def test_remove_outliers_clip(self):
        self.processor.outlier_method = "clip"
        self.processor.outlier_limits = [0.0, 0.8]
        df = pd.DataFrame({"a": [1, 2, 100], "b": [1, 2, 3]})
        cleaned = self.processor._remove_outliers(df.copy())
        self.assertLessEqual(cleaned["a"].max(), df["a"].quantile(0.8))

    def test_generate_labels(self):
        df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
        labeled = self.processor.generate_labels(
            df.copy(), forward_returns=1, threshold=0.5
        )
        self.assertIn("label", labeled.columns)
        self.assertEqual(len(labeled), 4)


if __name__ == "__main__":
    unittest.main()
