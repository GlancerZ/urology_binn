import numpy as np
import unittest

from maps_discrete.endpoint import discrete_targets, fixed_horizon_status, years_from_days
from maps_discrete.final_evaluation import _recovery
from maps_discrete.metrics import continuous_nri_idi, decision_curve, harrell_c_index


class MetricTests(unittest.TestCase):
    def test_protocol_year_uses_365_days(self):
        self.assertEqual(years_from_days(np.array([365.0]))[0], 1.0)

    def test_early_censoring_is_not_a_fixed_horizon_control(self):
        known, y = fixed_horizon_status(np.array([2.0, 10.0, 4.0]), np.array([0, 0, 1]), 10)
        self.assertEqual(known.tolist(), [False, True, True])
        self.assertEqual(y.tolist(), [0, 0, 1])

    def test_discrete_target_event_and_censor_masks(self):
        target, mask = discrete_targets(np.array([1.4, 3.2]), np.array([1, 0]), 4)
        self.assertEqual(target[0].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(mask[0].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(mask[1].tolist(), [1.0, 1.0, 1.0, 0.0])

    def test_harrell_c_perfect_order(self):
        duration = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 0])
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        self.assertEqual(harrell_c_index(duration, event, risk), 1.0)

    def test_nri_idi_and_dca_direction(self):
        duration = np.array([2.0, 3.0, 10.0, 10.0])
        event = np.array([1, 1, 0, 0])
        old = np.array([0.4, 0.4, 0.4, 0.4])
        new = np.array([0.9, 0.8, 0.2, 0.1])
        result = continuous_nri_idi(duration, event, old, new, 10)
        self.assertEqual(result["nri"], 2.0)
        self.assertGreater(result["idi"], 0)
        curve = decision_curve(duration, event, new, np.array([0.5]), 10)
        self.assertGreater(curve.loc[0, "model_net_benefit"], 0)

    def test_direct_student_teacher_recovery_ratio(self):
        self.assertAlmostEqual(_recovery("auc", 0.72, 0.80), 0.90)
        self.assertAlmostEqual(_recovery("brier", 0.05, 0.04), 0.80)


if __name__ == "__main__":
    unittest.main()
