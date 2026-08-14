#!/usr/bin/env python3
"""
These are the thorough unit tests for the ejtk helpers
"""

import numpy as np
import pytest
from circadian_cell_painting.osc_detection._ejtk_cycle_tools import (
    compute_best_tau,
    init_ref_cosines,
    _build_dist_params,
    hlm,
    _s_to_pvalue,
)

# --- Testing HLM ---


class TestHLM:
    """
    Tests the Hodges-Lehmann estimator
    """

    # -- basic correctness

    def test_single_value(self):
        """hlm of a single value is that value"""
        assert hlm(np.array([7.0])) == pytest.approx(7.0)

    def test_two_values(self):
        """hlm of two values is their mean"""
        assert hlm(np.array([2.0, 4.0])) == pytest.approx(3.0)

    def test_symmetric_array_equals_mean(self):
        """For symmetric distributions hlm == mean == median"""
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hlm(z) == pytest.approx(3.0)

    def test_constant_array(self):
        z = np.array([5.0, 5.0, 5.0, 5.0])
        assert hlm(z) == pytest.approx(5.0)

    def test_zero_array(self):
        z = np.array([0.0, 0.0, 0.0])
        assert hlm(z) == pytest.approx(0.0)

    def test_negative_values(self):
        """Symmetric around 0 → hlm = 0"""
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert hlm(z) == pytest.approx(0.0, abs=1e-10)

    def test_all_negative(self):
        z = np.array([-5.0, -3.0, -1.0])
        assert hlm(z) < 0

    # -- known pairwise avg examples

    def test_known_pairwise_example(self):
        """
        z = [1, 3, 5]
        All pairwise averages including self-pairs (lower triangle with diag):
        (1+1)/2=1, (1+3)/2=2, (3+3)/2=3, (1+5)/2=3, (3+5)/2=4, (5+5)/2=5
        median([1, 2, 3, 3, 4, 5]) = (3+3)/2 = 3
        """
        z = np.array([1.0, 3.0, 5.0])
        assert hlm(z) == pytest.approx(3.0)

    def test_known_asymmetric_example(self):
        """
        z = [1, 2, 4]
        Pairwise averages: 1, 1.5, 2, 2.5, 3, 4
        median = (2 + 2.5) / 2 = 2.25
        """
        z = np.array([1.0, 2.0, 4.0])
        assert hlm(z) == pytest.approx(2.25)

    # -- robustness

    def test_robust_to_outlier(self):
        """
        hlm should be more resistant to outliers than mean.
        z = [1, 2, 3, 4, 100]  mean=22, median=3
        hlm should be closer to median than mean
        """
        z = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = hlm(z)
        assert abs(result - 3.0) < abs(
            result - 22.0
        ), "hlm should be closer to median than mean for skewed data"

    def test_large_outlier_bounded(self):
        z = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
        assert hlm(z) < 100.0, "hlm should not be dominated by single outlier"

    # -- return type

    def test_returns_float(self):
        assert isinstance(hlm(np.array([1.0, 2.0, 3.0])), float)

    def test_result_is_finite(self):
        z = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isfinite(hlm(z))

    def test_large_array(self):
        """Should not crash on large arrays"""
        z = np.random.default_rng(0).normal(0, 1, 100)
        result = hlm(z)
        assert np.isfinite(result)

    # -- edge cases

    def test_empty_array_returns_zero(self):
        """Empty or all-non-finite array should return 0.0 gracefully"""
        result = hlm(np.array([np.nan, np.inf]))
        assert result == 0.0

    def test_ignores_inf(self):
        """Non-finite values should be excluded"""
        z_clean = np.array([1.0, 2.0, 3.0])
        z_dirty = np.array([1.0, 2.0, 3.0, np.inf])
        assert hlm(z_clean) == pytest.approx(hlm(z_dirty), abs=0.5)

    def test_order_invariant(self):
        """hlm should not depend on order of input"""
        z1 = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        z2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hlm(z1) == pytest.approx(hlm(z2))

    def test_scale_equivariant(self):
        """hlm(c*z) == c * hlm(z) for c > 0"""
        z = np.array([1.0, 2.0, 3.0, 4.0])
        assert hlm(2.0 * z) == pytest.approx(2.0 * hlm(z))

    def test_location_equivariant(self):
        """hlm(z + c) == hlm(z) + c"""
        z = np.array([1.0, 2.0, 3.0, 4.0])
        assert hlm(z + 10.0) == pytest.approx(hlm(z) + 10.0)


class TestBuildDistParams:
    """
    Tests for _build_dist_params.
    Builds the JTK null distribution for a given number of timepoints.
    """

    # ── output structure

    def test_returns_dict(self):
        dp = _build_dist_params(12)
        assert isinstance(dp, dict)

    def test_always_has_exact_key(self):
        dp = _build_dist_params(12)
        assert "exact" in dp

    def test_always_has_max_key(self):
        dp = _build_dist_params(12)
        assert "max" in dp

    def test_exact_path_has_cp(self, dp_small):
        if dp_small["exact"]:
            assert dp_small["cp"] is not None
            assert isinstance(dp_small["cp"], np.ndarray)

    def test_normal_path_has_sdv_exv(self, dp_large):
        if not dp_large["exact"]:
            assert "sdv" in dp_large
            assert "exv" in dp_large
            assert dp_large["sdv"] > 0
            assert dp_large["exv"] > 0

    # ── max statistic

    def test_max_positive(self, dp_small):
        assert dp_small["max"] > 0

    def test_max_formula(self):
        """max = (n^2 - n) / 2 for reps=1"""
        n = 10
        dp = _build_dist_params(n)
        assert dp["max"] == (n**2 - n) // 2

    def test_more_timepoints_larger_max(self):
        dp10 = _build_dist_params(10)
        dp20 = _build_dist_params(20)
        assert dp20["max"] > dp10["max"]

    def test_max_is_integer(self, dp_small):
        assert isinstance(dp_small["max"], int)

    # ── cp array properties

    def test_cp_length(self, dp_small):
        """cp should have length 2*max + 1"""
        if dp_small["exact"]:
            assert len(dp_small["cp"]) == 2 * dp_small["max"] + 1

    def test_cp_first_value_is_one(self, dp_small):
        """cp[0] = P(S >= 0) = 1.0 (all permutations have S >= 0 by symmetry)"""
        if dp_small["exact"]:
            assert dp_small["cp"][0] == pytest.approx(1.0, abs=1e-6)

    def test_cp_values_in_0_1(self, dp_small):
        if dp_small["exact"]:
            assert np.all(dp_small["cp"] >= -1e-10)
            assert np.all(dp_small["cp"] <= 1.0 + 1e-10)

    def test_cp_non_increasing(self, dp_small):
        """Upper tail CDF should be non-increasing"""
        if dp_small["exact"]:
            diffs = np.diff(dp_small["cp"])
            assert np.all(
                diffs <= 1e-10
            ), "cp should be non-increasing (upper tail CDF)"

    def test_cp_last_value_near_zero(self, dp_small):
        """cp[-1] = P(S >= max) should be very small"""
        if dp_small["exact"]:
            assert dp_small["cp"][-1] < 0.1

    # ── determinism

    def test_deterministic(self):
        dp1 = _build_dist_params(12)
        dp2 = _build_dist_params(12)
        assert dp1["max"] == dp2["max"]
        assert dp1["exact"] == dp2["exact"]
        if dp1["exact"]:
            np.testing.assert_array_almost_equal(dp1["cp"], dp2["cp"])

    # ── various n values

    def test_n_2(self):
        """Minimum meaningful n"""
        dp = _build_dist_params(2)
        assert dp["max"] == 1
        assert dp is not None

    def test_n_5(self):
        dp = _build_dist_params(5)
        assert dp["max"] == (25 - 5) // 2

    def test_n_48(self):
        dp = _build_dist_params(48)
        assert dp["max"] > 0
        assert dp is not None

    def test_n_100_does_not_crash(self):
        dp = _build_dist_params(100)
        assert dp is not None
        assert dp["max"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestSToP
# ══════════════════════════════════════════════════════════════════════════════


class TestSToP:
    """
    Tests for _s_to_pvalue.
    Converts Kendall S statistic to two-tailed p-value.
    """

    # ── boundary values

    def test_zero_s_returns_one(self, dp_small):
        """S=0 means no concordance — p=1"""
        assert _s_to_pvalue(0.0, dp_small) == 1.0

    def test_max_s_small_pvalue(self, dp_small):
        """Maximum possible S should be highly significant"""
        p = _s_to_pvalue(float(dp_small["max"]), dp_small)
        assert p < 0.05, f"Max S should give p < 0.05, got {p:.4f}"

    def test_max_s_very_small(self, dp_medium):
        """For larger n, max S should be extremely significant"""
        p = _s_to_pvalue(float(dp_medium["max"]), dp_medium)
        assert p < 0.001

    # ── valid range

    def test_pvalue_in_0_1_for_zero(self, dp_small):
        assert 0.0 <= _s_to_pvalue(0.0, dp_small) <= 1.0

    def test_pvalue_in_0_1_for_small_s(self, dp_small):
        assert 0.0 <= _s_to_pvalue(5.0, dp_small) <= 1.0

    def test_pvalue_in_0_1_for_max(self, dp_small):
        assert 0.0 <= _s_to_pvalue(float(dp_small["max"]), dp_small) <= 1.0

    def test_pvalue_in_0_1_for_many_values(self, dp_medium):
        for S in np.linspace(0, dp_medium["max"], 20):
            p = _s_to_pvalue(float(S), dp_medium)
            assert 0.0 <= p <= 1.0, f"p={p:.4f} out of range for S={S:.1f}"

    # ── monotonicity

    def test_larger_s_smaller_p(self, dp_medium):
        """Larger |S| should give smaller or equal p"""
        S_vals = [10.0, 30.0, 60.0, 100.0]
        p_vals = [_s_to_pvalue(S, dp_medium) for S in S_vals]
        for i in range(len(p_vals) - 1):
            assert (
                p_vals[i + 1] <= p_vals[i] + 1e-10
            ), f"p should decrease as S increases: p({S_vals[i]})={p_vals[i]:.4f}, p({S_vals[i+1]})={p_vals[i+1]:.4f}"

    # ── two-tailed symmetry

    def test_negative_s_same_as_positive(self, dp_medium):
        """Two-tailed: p(-S) should equal p(S)"""
        for S in [10.0, 50.0, 100.0]:
            assert _s_to_pvalue(-S, dp_medium) == pytest.approx(
                _s_to_pvalue(S, dp_medium), abs=1e-10
            )

    # ── exact vs normal approximation ────────────────────────────────────────

    def test_normal_approx_valid_range(self):
        """Normal approximation path should return valid p-value"""
        dp = {
            "exact": False,
            "max": 1000,
            "sdv": 100.0,
            "exv": 500.0,
            "cp": None,
        }
        p = _s_to_pvalue(600.0, dp)
        assert 0.0 < p <= 1.0

    def test_normal_approx_zero_s_returns_one(self):
        dp = {
            "exact": False,
            "max": 1000,
            "sdv": 100.0,
            "exv": 500.0,
            "cp": None,
        }
        assert _s_to_pvalue(0.0, dp) == 1.0

    def test_exact_and_normal_agree_roughly(self):
        """
        For moderate n where both paths might be available,
        exact and normal approx should give similar results.
        Force normal approx and compare to exact.
        """
        dp_exact = _build_dist_params(20)
        dp_normal = {
            "exact": False,
            "max": dp_exact["max"],
            "sdv": np.sqrt((20**2 * (2 * 20 + 3)) / 72),
            "exv": dp_exact["max"] / 2,
            "cp": None,
        }
        S = float(dp_exact["max"]) * 0.5
        p_exact = _s_to_pvalue(S, dp_exact)
        p_normal = _s_to_pvalue(S, dp_normal)
        assert (
            abs(p_exact - p_normal) < 0.1
        ), f"Exact ({p_exact:.4f}) and normal ({p_normal:.4f}) too different"

    # ── returns float

    def test_returns_float(self, dp_small):
        result = _s_to_pvalue(10.0, dp_small)
        assert isinstance(result, float)

    def test_result_is_finite(self, dp_small):
        result = _s_to_pvalue(10.0, dp_small)
        assert np.isfinite(result)


# ══════════════════════════════════════════════════════════════════════════════
# TestInitRefCosines
# ══════════════════════════════════════════════════════════════════════════════


class TestInitRefCosines:
    """
    Tests for init_ref_cosines.
    Precomputes rank-based reference cosines for all (period, lag) combos.
    """

    # ── output structure

    def test_returns_dict(self, periods, ref_48tp):
        assert isinstance(ref_48tp, dict)

    def test_all_periods_present(self, periods, ref_48tp):
        assert len(ref_48tp) == len(periods)

    def test_keys_are_integer_indices(self, periods, ref_48tp):
        for i in range(len(periods)):
            assert i in ref_48tp

    def test_each_value_is_tuple_of_two(self, periods, ref_48tp):
        for pi in range(len(periods)):
            val = ref_48tp[pi]
            assert isinstance(val, tuple)
            assert len(val) == 2

    # ── cgoosv shape

    def test_cgoosv_shape(self, periods, ref_48tp):
        """cgoosv should be (n_pairs, period) where n_pairs = n*(n-1)/2"""
        n = 48
        n_pairs = n * (n - 1) // 2
        for pi, period in enumerate(periods):
            cgoosv, _ = ref_48tp[pi]
            assert cgoosv.shape == (
                n_pairs,
                period,
            ), f"period={period}: expected ({n_pairs},{period}), got {cgoosv.shape}"

    def test_cgoosv_shape_small_n(self, periods):
        ref = init_ref_cosines(periods, 24)
        n_pairs = 24 * 23 // 2
        for pi, period in enumerate(periods):
            cgoosv, _ = ref[pi]
            assert cgoosv.shape == (n_pairs, period)

    # ── signcos shape

    def test_signcos_shape(self, periods, ref_48tp):
        """signcos should be (n_full_cycle_points, period) — v3.1 full cycles"""
        n = 48
        for pi, period in enumerate(periods):
            _, signcos = ref_48tp[pi]
            n_full = (n // period) * period
            assert signcos.shape == (
                n_full,
                period,
            ), f"period={period}: expected ({n_full},{period}), got {signcos.shape}"

    # ── value constraints

    def test_cgoosv_values_in_minus1_0_1(self, periods, ref_48tp):
        """cgoosv entries are sign differences: only -1, 0, 1"""
        for pi in range(len(periods)):
            cgoosv, _ = ref_48tp[pi]
            unique = np.unique(cgoosv)
            assert all(
                v in [-1.0, 0.0, 1.0] for v in unique
            ), f"cgoosv has unexpected values: {unique}"

    def test_signcos_values_in_minus1_0_1(self, periods, ref_48tp):
        """signcos entries are sign of cosine: only -1, 0, 1"""
        for pi in range(len(periods)):
            _, signcos = ref_48tp[pi]
            unique = np.unique(signcos)
            assert all(
                v in [-1.0, 0.0, 1.0] for v in unique
            ), f"signcos has unexpected values: {unique}"

    def test_cgoosv_antisymmetric_property(self, periods):
        """
        cgoosv is built from lower triangle of sign(outer(cos_r, cos_r)).
        Each column should have both positive and negative values
        (cosine has both ascending and descending regions).
        """
        ref = init_ref_cosines(periods, 48)
        for pi in range(len(periods)):
            cgoosv, _ = ref[pi]
            for lag in range(cgoosv.shape[1]):
                col = cgoosv[:, lag]
                assert np.any(
                    col > 0
                ), f"period_idx={pi}, lag={lag}: no positive values"
                assert np.any(
                    col < 0
                ), f"period_idx={pi}, lag={lag}: no negative values"

    # ── lag variation

    def test_different_lags_different_cgoosv(self, periods, ref_48tp):
        """Each lag should produce a different reference cosine"""
        for pi, period in enumerate(periods):
            cgoosv, _ = ref_48tp[pi]
            for lag1 in range(period):
                for lag2 in range(lag1 + 1, period):
                    assert not np.array_equal(
                        cgoosv[:, lag1], cgoosv[:, lag2]
                    ), f"period={period}: lags {lag1} and {lag2} have identical cgoosv"

    # def test_different_lags_different_signcos(self, periods, ref_48tp):
    #     for pi, period in enumerate(periods):
    #         _, signcos = ref_48tp[pi]
    #         for lag1 in range(period):
    #             for lag2 in range(lag1 + 1, period):
    #                 assert not np.array_equal(
    #                     signcos[:, lag1], signcos[:, lag2]
    #                 ), f"period={period}: lags {lag1} and {lag2} have identical signcos"

    # ── determinism

    def test_deterministic(self, periods):
        ref1 = init_ref_cosines(periods, 48)
        ref2 = init_ref_cosines(periods, 48)
        for pi in range(len(periods)):
            np.testing.assert_array_equal(ref1[pi][0], ref2[pi][0])
            np.testing.assert_array_equal(ref1[pi][1], ref2[pi][1])

    # ── different n_timepoints

    def test_different_n_different_shape(self, periods):
        ref24 = init_ref_cosines(periods, 24)
        ref48 = init_ref_cosines(periods, 48)
        # n_pairs grows with n
        assert ref48[0][0].shape[0] > ref24[0][0].shape[0]

    def test_single_period(self):
        ref = init_ref_cosines([12], 48)
        assert len(ref) == 1
        cgoosv, signcos = ref[0]
        n_pairs = 48 * 47 // 2
        assert cgoosv.shape == (n_pairs, 12)

    # ── dtype

    def test_cgoosv_dtype_float(self, periods, ref_48tp):
        for pi in range(len(periods)):
            cgoosv, _ = ref_48tp[pi]
            assert cgoosv.dtype in [np.int32, np.int64]

    def test_signcos_dtype_float(self, periods, ref_48tp):
        for pi in range(len(periods)):
            _, signcos = ref_48tp[pi]
            assert signcos.dtype in [np.float64, float]
