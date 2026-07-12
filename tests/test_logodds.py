"""Log-odds fusion: the properties the arithmetic mean lacks."""
import numpy as np
import pytest

from negaverse.ig.logodds_fusion import (
    LogOddsStack, logit, logodds_fuse, logodds_pool, sigmoid,
)
from negaverse.schema import StreamScore


def _ss(name, value, veto=False):
    return StreamScore(name, value=value, veto=veto)


def test_agreement_reinforces_not_dilutes():
    # the mean's failure case: two "safe" streams at 0.75 and 1.0.
    # mean -> 0.875 (< the stronger stream). log-odds pool -> above both.
    conf = logodds_fuse([_ss("a", 0.75), _ss("b", 0.99)]).confidence
    assert conf > 0.99
    # a third agreeing safe stream raises confidence further (mean would keep diluting)
    conf3 = logodds_fuse([_ss("a", 0.75), _ss("b", 0.99), _ss("c", 0.8)]).confidence
    assert conf3 > conf


def test_abstention_is_neutral():
    both = logodds_fuse([_ss("a", 0.8), _ss("b", 0.7)]).confidence
    with_abstain = logodds_fuse([_ss("a", 0.8), _ss("b", 0.7), _ss("c", None)]).confidence
    assert both == with_abstain


def test_contradiction_cancels():
    # symmetric disagreement around 0.5 -> back to the prior
    conf = logodds_fuse([_ss("a", 0.9), _ss("b", 0.1)]).confidence
    assert conf == pytest.approx(0.5, abs=1e-3)


def test_veto_drops():
    f = logodds_fuse([_ss("a", 0.9), _ss("v", None, veto=True)])
    assert f.vetoed and f.confidence is None


def test_all_abstain_returns_prior():
    assert logodds_fuse([_ss("a", None), _ss("b", None)]).confidence == pytest.approx(0.5)


def test_pool_matches_manual():
    L = logodds_pool({"a": 0.8, "b": 0.6, "c": None})
    assert L == pytest.approx(logit(0.8) + logit(0.6))
    assert sigmoid(L) == pytest.approx(logodds_fuse([_ss("a", 0.8), _ss("b", 0.6)]).confidence, abs=1e-4)


def test_weights_scale_evidence():
    strong = logodds_pool({"a": 0.8}, weights={"a": 2.0})
    weak = logodds_pool({"a": 0.8}, weights={"a": 1.0})
    assert strong == pytest.approx(2 * weak)


def test_stack_calibrates_and_lcb_is_a_floor():
    rng = np.random.default_rng(0)
    n = 400
    # a genuinely predictive logit feature + a redundant near-copy (needs shrinkage)
    z0 = rng.normal(size=n)
    Z = np.column_stack([z0, z0 + 0.01 * rng.normal(size=n)])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.5 * z0))).astype(int)
    stack = LogOddsStack(l2=1.0).fit(Z, y)
    p = stack.proba(Z)
    lcb = stack.lcb(Z, alpha=0.05)
    # discriminates
    assert np.corrcoef(p, y)[0, 1] > 0.3
    # LCB never exceeds the point estimate
    assert np.all(lcb <= p + 1e-9)
    # LCB is a strict floor somewhere (nonzero coefficient uncertainty)
    assert np.any(lcb < p - 1e-6)
