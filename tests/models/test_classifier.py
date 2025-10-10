import unittest
import pandas as pd
import numpy as np

from quanttradeai.models.classifier import MomentumClassifier


class TestMomentumClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 20
        self.df = pd.DataFrame(
            {
                "Open": np.random.rand(n),
                "High": np.random.rand(n),
                "Low": np.random.rand(n),
                "Close": np.random.rand(n),
                "Volume": np.random.rand(n),
                "feature1": np.random.rand(n),
                "feature2": np.random.rand(n),
                "forward_returns": np.random.rand(n),
                "label": np.random.randint(0, 2, n),
            }
        )
        self.model = MomentumClassifier("config/model_config.yaml")
        self.X, self.y = self.model.prepare_data(self.df)

    def test_train_and_predict(self):
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
