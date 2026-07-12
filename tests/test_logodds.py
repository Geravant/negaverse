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


def test_pipeline_calibrator_unit():
    """PipelineCalibrator: fit on labeled sub-score dicts, transform a candidate."""
    from negaverse.ig.logodds_fusion import PipelineCalibrator
    rng = np.random.default_rng(0)
    names = ["structured", "topology"]
    subs, y = [], []
    for _ in range(200):                       # negatives: high stream values
        subs.append({"structured": float(rng.uniform(0.6, 1.0)),
                     "topology": float(rng.uniform(0.6, 1.0))}); y.append(1)
    for _ in range(200):                       # positives: low stream values
        subs.append({"structured": float(rng.uniform(0.0, 0.4)),
                     "topology": None}); y.append(0)   # abstention tolerated
    cal = PipelineCalibrator(names, l2=0.2).fit(subs, y)
    p_neg, lcb_neg = cal.transform({"structured": 0.9, "topology": 0.9})
    p_pos, lcb_pos = cal.transform({"structured": 0.1, "topology": 0.1})
    assert 0.0 <= p_pos <= 1.0 and 0.0 <= p_neg <= 1.0
    assert p_neg > p_pos                       # ranks a clean pair above a link-like one
    assert lcb_neg <= p_neg and lcb_pos <= p_pos


def test_pipeline_calibration_stage_end_to_end():
    """run_pipeline(calibrate=True) emits a calibrated confidence + LCB per record."""
    from negaverse import run_pipeline, PipelineConfig
    from negaverse.io import load_sars_cov2_graph
    graph = load_sars_cov2_graph()
    viral = list(graph.nodes_of_type("viral"))
    host = list(graph.nodes_of_type("host"))
    edges = {frozenset(e) for e in graph.g.edges()}
    gold = []                                  # admissible non-edges as stand-in gold negatives
    for a in viral:
        for b in host:
            if frozenset((a, b)) not in edges:
                gold.append((a, b))
            if len(gold) >= 60:
                break
        if len(gold) >= 60:
            break
    base = PipelineConfig(n_eval=120, n_train=120, seed=0, match_on_type="viral", calibrate=False)
    cfg = PipelineConfig(n_eval=120, n_train=120, seed=0, match_on_type="viral",
                         calibrate=True, calibration_gold=gold)
    r_base = run_pipeline(graph, base)
    r_cal = run_pipeline(graph, cfg)
    # calibration populates SEPARATE authorising fields, both present and ordered
    assert all(rec.confidence_calibrated is not None and rec.confidence_lcb is not None
               for rec in r_cal.records)
    assert all(0.0 <= rec.confidence_lcb <= rec.confidence_calibrated <= 1.0 for rec in r_cal.records)
    assert len({round(rec.confidence_calibrated, 3) for rec in r_cal.records}) > 1   # not a constant
    # base run (calibrate=False) leaves the calibrated fields unset
    assert all(rec.confidence_calibrated is None and rec.confidence_lcb is None
               for rec in r_base.records)
    # the calibrated estimate is a DISTINCT axis from the selection-driving confidence
    assert any(abs(rec.confidence_calibrated - rec.confidence) > 1e-6 for rec in r_cal.records)


def test_gold_recall_scale_robust():
    """gold_recall's default (quantile) path is invariant to a monotone rescale of
    confidence — the property that makes it safe under calibration."""
    from negaverse import eval as ev
    from negaverse.schema import NegativeRecord

    def rec(u, v, c):
        return NegativeRecord(u=u, v=v, mode="train", confidence=c, hardness=0.5,
                              streams={}, provenance={})
    # 10 records; the two golds sit in the upper half of the confidence pool
    confs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    records = [rec(f"g{i}", f"h{i}", c) for i, c in enumerate(confs)]
    gold = {frozenset(("g7", "h7")), frozenset(("g9", "h9"))}   # confidences 0.80, 0.95

    base = ev.gold_recall(records, gold)                        # default: pool-median quantile
    assert base["golds_in_pool"] == 2
    assert base["recall_at_threshold"] == 1.0                  # both golds above the median
    assert base["threshold_kind"].startswith("pool_q")

    # monotone rescale of EVERY confidence must not change the quantile-based recall
    for r in records:
        r.confidence = r.confidence ** 2                       # monotone, changes the scale
    rescaled = ev.gold_recall(records, gold)
    assert rescaled["recall_at_threshold"] == base["recall_at_threshold"]

    # explicit absolute threshold still supported and behaves absolutely
    for r, c in zip(records, confs):
        r.confidence = c
    hi = ev.gold_recall(records, gold, conf_threshold=0.85)
    assert hi["threshold_kind"] == "abs>=0.85" and hi["high_confidence"] == 1   # only 0.95
