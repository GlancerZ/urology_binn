import unittest

import numpy as np
import torch
import torch.nn.functional as F

from maps_discrete.mixture_cure import OrdinalTimingHead, SoftmaxTimingHead, mixture_nll
from maps_discrete.mixture_pipeline import onset_distribution, timing_metrics, timing_summary


class MixtureCureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logit = torch.randn(4)
        self.log_f = torch.log_softmax(torch.randn(4, 5), dim=1)

    def test_complete_follow_up_is_incidence_bce_plus_case_cross_entropy(self):
        targets = torch.zeros(4, 5)
        targets[0, 1] = 1.0
        targets[1, 4] = 1.0
        mask = torch.ones(4, 5)
        mask[0, 2:] = 0.0
        nll = mixture_nll(self.logit, self.log_f, targets, mask)
        bce = F.binary_cross_entropy_with_logits(self.logit, torch.tensor([1.0, 1.0, 0.0, 0.0]), reduction="none")
        case_ce = torch.stack([-self.log_f[0, 1], -self.log_f[1, 4], torch.tensor(0.0), torch.tensor(0.0)])
        torch.testing.assert_close(nll, bce + case_ce)

    def test_early_censoring_uses_latency_survival(self):
        targets = torch.zeros(1, 5)
        mask = torch.zeros(1, 5)
        mask[0, :2] = 1.0
        nll = mixture_nll(self.logit[:1], self.log_f[:1], targets, mask)
        pi = torch.sigmoid(self.logit[0])
        onset = self.log_f[0].exp()
        torch.testing.assert_close(nll[0], -torch.log((1 - pi) + pi * (1 - onset[:2].sum())))

    def test_ordinal_head_is_a_distribution_and_score_moves_onset_earlier(self):
        head = OrdinalTimingHead(n_tokens=3, n_intervals=5, embed_dim=4)
        reference = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        head.initialize(reference)
        # Equal token values v give score = gamma * v = v, whatever the attention.
        tokens = torch.tensor([[0.0] * 3, [-2.0] * 3, [2.0] * 3])
        log_probability, score, _ = head(tokens)
        probability = log_probability.exp().detach()
        torch.testing.assert_close(probability.sum(dim=1), torch.ones(3))
        np.testing.assert_allclose(probability[0].numpy(), reference, atol=1e-3)
        expected_year = (probability * (torch.arange(5) + 0.5)).sum(dim=1)
        self.assertGreater(float(expected_year[1]), float(expected_year[0]))
        self.assertLess(float(expected_year[2]), float(expected_year[0]))

    def test_risk_coupled_ordinal_head_moves_onset_earlier_with_risk(self):
        head = OrdinalTimingHead(n_tokens=3, n_intervals=5, embed_dim=4, use_tokens=False, use_risk=True)
        head.initialize(np.full(5, 0.2))
        with torch.no_grad():
            head.coupling.fill_(1.0)
        risk = torch.tensor([-2.0, 0.0, 2.0])
        log_probability, score, attention = head(torch.randn(3, 3), risk)
        torch.testing.assert_close(score, risk)
        self.assertIsNone(attention)
        expected_year = (log_probability.exp() * (torch.arange(5) + 0.5)).sum(dim=1)
        self.assertTrue(bool((expected_year[:-1] > expected_year[1:]).all()))

    def test_softmax_head_starts_at_the_reference_distribution(self):
        head = SoftmaxTimingHead(n_tokens=3, n_intervals=4)
        reference = np.array([0.1, 0.2, 0.3, 0.4])
        head.initialize(reference)
        log_probability, _, _ = head(torch.randn(2, 3))
        np.testing.assert_allclose(log_probability.exp().detach().numpy(), np.tile(reference, (2, 1)), atol=1e-6)

    def test_onset_distribution_and_summary(self):
        np.testing.assert_allclose(onset_distribution(np.array([[0.1, 0.3, 0.4]])), [[0.25, 0.5, 0.25]])
        summary = timing_summary(np.full((1, 10), 0.1)).iloc[0]
        self.assertAlmostEqual(summary["expected_onset_year"], 5.0)
        self.assertAlmostEqual(summary["median_onset_year"], 5.0)
        self.assertAlmostEqual(summary["onset_year_q10"], 1.0)
        self.assertAlmostEqual(summary["onset_year_q90"], 9.0)
        self.assertAlmostEqual(summary["p_onset_0_3y"], 0.3)
        self.assertAlmostEqual(summary["p_onset_3_7y"], 0.4)

    def test_perfect_timing_prediction(self):
        duration = np.array([0.5, 2.5, 8.5, 9.5, 10.0])
        event = np.array([1, 1, 1, 1, 0])
        onset = np.zeros((5, 10))
        onset[np.arange(5), [0, 2, 8, 9, 5]] = 1.0
        result = timing_metrics(duration, event, onset, np.full(10, 0.1), horizon=10)
        self.assertEqual(result["timing_cases"], 4)
        self.assertEqual(result["timing_concordance"], 1.0)
        self.assertEqual(result["early_vs_late_auc"], 1.0)
        self.assertEqual(result["interval80_coverage"], 1.0)
        self.assertGreater(result["timing_log_score_gain"], 0)


if __name__ == "__main__":
    unittest.main()
