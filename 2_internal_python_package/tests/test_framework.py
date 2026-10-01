from pathlib import Path
import unittest

import pandas as pd

from maps_discrete.config import load_config
from maps_discrete.guard import ProtocolNotApprovedError, require_approved
from maps_discrete.selection import UnsafeSeedSelectionError, rank_seeds
from maps_discrete.splits import audit_split_manifest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "3_frozen_configs" / "analysis_config.toml"


class FrameworkTests(unittest.TestCase):
    def test_formal_protocol_is_approved(self):
        config = load_config(CONFIG)
        self.assertTrue(config.approved)
        require_approved(config)

    def test_test_based_seed_selection_is_blocked(self):
        results = pd.DataFrame({"training_seed": [1, 2], "auc_10y": [0.7, 0.8]})
        with self.assertRaises(UnsafeSeedSelectionError):
            rank_seeds(
                results,
                metric="auc_10y",
                n=1,
                higher_is_better=True,
                dataset_role="test",
            )

    def test_explicit_exploratory_test_ranking_is_available(self):
        results = pd.DataFrame({"training_seed": [1, 2], "auc_10y": [0.7, 0.8]})
        selected = rank_seeds(
            results,
            metric="auc_10y",
            n=1,
            higher_is_better=True,
            dataset_role="test",
            analysis_role="exploratory",
            allow_test_based_selection=True,
        )
        self.assertEqual(selected, [2])

    def test_existing_split_manifest_is_valid(self):
        config = load_config(CONFIG)
        self.assertEqual(config.test_holdout_seed, 2026)
        self.assertEqual(config.train_validation_split_seed, 10)
        audit = audit_split_manifest(config.raw["data"]["source_split_manifest"])
        self.assertEqual(audit.n, 17259)
        self.assertEqual(audit.train_n + audit.val_n + audit.test_n, audit.n)


if __name__ == "__main__":
    unittest.main()
