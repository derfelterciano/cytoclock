#!/usr/bin/env python3
import pytest
import numpy as np
from cytoclock.osc_detection._ejtk_cycle_tools import compute_best_tau
from cytoclock.osc_detection import eJTK_CYCLE


class TestComputeBestTau:
    def test_returns_6_tuple(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        result = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert len(result) == 6

    def test_unpacks_correctly(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        tau, raw_p, adj_p, period, lag, amp = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert isinstance(tau, float)
        assert isinstance(raw_p, float)
        assert isinstance(adj_p, float)
        assert isinstance(period, (int, np.integer))
        assert isinstance(lag, float)
        assert isinstance(amp, float)

    def test_tau_in_0_1(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        tau, _, _, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert 0.0 <= tau <= 1.0

    def test_raw_pvalue_in_0_1(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, raw_p, _, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert 0.0 <= raw_p <= 1.0

    def test_adj_pvalue_in_0_1(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, _, adj_p, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert 0.0 <= adj_p <= 1.0

    def test_adj_pvalue_gte_raw_pvalue(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        """Bonferroni-adjusted p should always be >= raw p"""
        _, raw_p, adj_p, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert adj_p >= raw_p

    def test_period_in_tested_range(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, _, _, period, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert period in periods

    def test_lag_within_period(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, _, _, period, lag, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert 0 <= lag < period

    def test_amplitude_nonnegative(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, _, _, _, _, amp = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert amp >= 0.0

    def test_clean_cosine_high_tau(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        tau, _, _, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert tau > 0.5, f"Expected tau > 0.5 for clean cosine, got {tau:.3f}"

    def test_clean_cosine_low_pvalue(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, _, adj_p, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert (
            adj_p < 0.05
        ), f"Expected significant adj p for clean cosine, got {adj_p:.4f}"

    def test_clean_cosine_amplitude_positive(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        _, _, _, _, _, amp = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert amp > 0.0

    def test_flat_signal_zero_amplitude(
        self, flat_signal, periods, ref_cosines_48, dist_params_48
    ):
        _, _, _, _, _, amp = compute_best_tau(
            flat_signal, periods, ref_cosines_48, dist_params_48
        )
        assert amp == pytest.approx(0.0, abs=1e-6)

    def test_flat_signal_tau_zero(
        self, flat_signal, periods, ref_cosines_48, dist_params_48
    ):
        tau, _, _, _, _, _ = compute_best_tau(
            flat_signal, periods, ref_cosines_48, dist_params_48
        )
        assert tau == pytest.approx(0.0, abs=1e-6)

    def test_random_noise_lower_tau_than_clean(
        self, periods, ref_cosines_48, dist_params_48
    ):
        rng = np.random.default_rng(7)
        clean_tau, _, _, _, _, _ = compute_best_tau(
            np.cos(2 * np.pi * np.arange(0, 96, 2) / 24),
            periods,
            ref_cosines_48,
            dist_params_48,
        )
        noise_taus = []
        for _ in range(10):
            noise = rng.normal(0, 1, 48)
            tau, _, _, _, _, _ = compute_best_tau(
                noise, periods, ref_cosines_48, dist_params_48
            )
            noise_taus.append(tau)
        assert clean_tau > np.mean(noise_taus)

    def test_phase_shifted_cosine_different_lag(
        self, periods, ref_cosines_48, dist_params_48
    ):
        t = np.arange(0, 96, 2, dtype=float)
        cos_a = np.cos(2 * np.pi * t / 24)
        cos_b = np.cos(2 * np.pi * t / 24 - np.pi / 2)

        _, _, _, _, lag_a, _ = compute_best_tau(
            cos_a, periods, ref_cosines_48, dist_params_48
        )
        _, _, _, _, lag_b, _ = compute_best_tau(
            cos_b, periods, ref_cosines_48, dist_params_48
        )

        assert lag_a != lag_b, "Phase-shifted cosines should recover different lags"

    def test_deterministic(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        r1 = compute_best_tau(clean_cosine_24h, periods, ref_cosines_48, dist_params_48)
        r2 = compute_best_tau(clean_cosine_24h, periods, ref_cosines_48, dist_params_48)
        assert r1 == r2

    def test_negated_signal_same_tau_magnitude(
        self, clean_cosine_24h, periods, ref_cosines_48, dist_params_48
    ):
        tau_pos, _, _, _, _, _ = compute_best_tau(
            clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        tau_neg, _, _, _, _, _ = compute_best_tau(
            -clean_cosine_24h, periods, ref_cosines_48, dist_params_48
        )
        assert tau_pos == pytest.approx(tau_neg, abs=1e-6)


class TestEjtkFitSingle:

    # ── basic structure

    def test_returns_dict(self, model, clean_cosine_24h, periods, ref_and_dist):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        assert isinstance(result, dict)

    def test_required_keys(self, model, clean_cosine_24h, periods, ref_and_dist):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        expected = {"p_value", "adj_p_jtk", "period", "phase_hours", "amplitude", "tau"}
        assert expected.issubset(result.keys())

    # ── value ranges

    def test_pvalue_in_0_1(self, model, clean_cosine_24h, periods, ref_and_dist):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        assert 0.0 < result["p_value"] <= 1.0

    def test_tau_in_0_1(self, model, clean_cosine_24h, periods, ref_and_dist):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        assert 0.0 <= result["tau"] <= 1.0

    def test_amplitude_nonnegative(
        self, model, clean_cosine_24h, periods, ref_and_dist
    ):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        assert result["amplitude"] >= 0.0

    def test_period_in_hours_range(
        self, model, clean_cosine_24h, periods, ref_and_dist
    ):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        # periods 10-14 tp * 2h interval = 20-28h
        assert 20.0 <= result["period"] <= 28.0

    # ── p-value floor

    def test_pvalue_never_zero(self, model, clean_cosine_24h, periods, ref_and_dist):
        """p_value should be floored at 1/n_perms, never exactly 0"""
        ref_cosines, dist_params = ref_and_dist
        n_perms = 20
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=n_perms,
            n_workers=2,
        )
        assert result["p_value"] >= 1.0 / n_perms

    # ── biological sanity

    def test_clean_rhythm_significant(
        self, model, clean_cosine_24h, periods, ref_and_dist
    ):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=100,
            n_workers=2,
        )
        assert (
            result["p_value"] < 0.05
        ), f"Clean cosine should be significant, got p={result['p_value']:.3f}"

    def test_random_noise_less_significant(
        self, model, random_signal, periods, ref_and_dist
    ):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=random_signal,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=100,
            n_workers=2,
        )
        assert (
            result["p_value"] > 0.01
        ), f"Random noise should not be highly significant, got p={result['p_value']:.3f}"

    def test_recovers_correct_period(
        self, model, clean_cosine_24h, periods, ref_and_dist
    ):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        assert (
            abs(result["period"] - 24.0) <= 4.0
        ), f"Expected period ~24h, got {result['period']:.1f}h"

    # ── error handling

    def test_returns_none_on_bad_input(self, model, periods, ref_and_dist):
        """Should return None (not raise) on invalid input due to try/except"""
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        result = model._ejtk_fit_single(
            timepoints=t,
            values=None,  # invalid input
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=20,
            n_workers=2,
        )
        assert result is None

    # ── reproducibility

    def test_same_seed_same_result(
        self, model, clean_cosine_24h, periods, ref_and_dist
    ):
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        r1 = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=50,
            n_workers=2,
            rng_seed=42,
        )
        r2 = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=50,
            n_workers=2,
            rng_seed=42,
        )
        assert r1["p_value"] == pytest.approx(r2["p_value"])
        assert r1["tau"] == pytest.approx(r2["tau"])

    def test_tau_deterministic_across_seeds(
        self, model, clean_cosine_24h, periods, ref_and_dist
    ):
        """tau/period should be identical regardless of rng_seed (observed stat, not permuted)"""
        ref_cosines, dist_params = ref_and_dist
        t = np.arange(48, dtype=float) * 2.0
        r1 = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=50,
            n_workers=2,
            rng_seed=1,
        )
        r2 = model._ejtk_fit_single(
            timepoints=t,
            values=clean_cosine_24h,
            periods=periods,
            ref_cosines=ref_cosines,
            dist_params=dist_params,
            interval=2.0,
            n_perms=50,
            n_workers=2,
            rng_seed=2,
        )
        assert r1["tau"] == pytest.approx(r2["tau"])
        assert r1["period"] == pytest.approx(r2["period"])
