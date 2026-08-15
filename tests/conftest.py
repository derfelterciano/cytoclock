#!/usr/bin/env python3
"""
This is a config for all our tests
"""

import pytest
from cytoclock.osc_detection._ejtk_cycle_tools import (
    compute_best_tau,
    init_ref_cosines,
    _build_dist_params,
    hlm,
    _s_to_pvalue,
)
from cytoclock.osc_detection import eJTK_CYCLE
import numpy as np


@pytest.fixture
def dp_small():
    """small n distribution params"""
    return _build_dist_params(12)


@pytest.fixture
def dp_medium():
    """medium n for dist params"""
    return _build_dist_params(24)


@pytest.fixture
def dp_large():
    """large n for dist params (this may trigger normal approx)"""
    return _build_dist_params(200)


@pytest.fixture
def ref_24tp(periods):
    return init_ref_cosines(periods, 24)


@pytest.fixture
def ref_48tp(periods):
    return init_ref_cosines(periods, 48)


@pytest.fixture
def periods():
    return list(range(10, 15))


@pytest.fixture
def ref_cosines_48(periods):
    return init_ref_cosines(periods, 48)


@pytest.fixture
def dist_params_48():
    return _build_dist_params(48)


@pytest.fixture
def clean_cosine_24h():
    """Clean 24h cosine @ 2h intervals, n=48"""
    t = np.arange(0, 96, 2, dtype=float)
    return np.cos(2 * np.pi * t / 24)


@pytest.fixture
def flat_signal():
    return np.ones(48)


@pytest.fixture
def model(periods):
    return eJTK_CYCLE(periods=periods, interval=2.0, n_perms=50, n_workers=2)


@pytest.fixture
def random_signal():
    return np.random.default_rng(0).normal(0, 1, 48)


@pytest.fixture
def ref_and_dist(model, clean_cosine_24h):
    """Forcing lazy init"""
    model._jtk_init(len(clean_cosine_24h))
    return model._ref_cosines, model._dist_params
