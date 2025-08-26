import unittest
import os
import yaml
import tempfile

from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor


class TestModelConfigValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "model.yaml")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_valid_model_config(self):
        config = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        loader = DataLoader(self.config_path)
        self.assertEqual(loader.symbols, ["AAPL"])

    def test_invalid_model_config_missing_symbols(self):
        config = {"data": {"start_date": "2020-01-01", "end_date": "2020-01-02"}}
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        with self.assertRaises(ValueError):
            DataLoader(self.config_path)


class TestFeatureConfigValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "features.yaml")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_valid_feature_config(self):
        config = {"pipeline": {"steps": ["a", "b"]}}
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        processor = DataProcessor(config_path=self.config_path)
        self.assertTrue(isinstance(processor.pipeline, list))

    def test_invalid_feature_config_missing_pipeline(self):
        config = {"price_features": []}
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        with self.assertRaises(ValueError):
            DataProcessor(config_path=self.config_path)


if __name__ == "__main__":
    unittest.main()
