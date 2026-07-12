"""Fisher-information (D-optimal) batch selection."""
import numpy as np

from negaverse.ig.fisher import greedy_fisher_design, logistic_fisher_weights


def _logdet(X, idx, ridge=1.0):
    d = X.shape[1]
    M = np.eye(d) * ridge
    for i in idx:
        M += np.outer(X[i], X[i])
    return np.linalg.slogdet(M)[1]


def test_picks_distinct_directions_over_duplicates():
    # 3 orthogonal directions, each duplicated 5x. D-optimal should take one from
    # each direction before ever taking a second copy of any.
    dirs = np.eye(3)
    X = np.repeat(dirs, 5, axis=0)              # 15 rows, rank 3
    sel = greedy_fisher_design(X, k=3, ridge=0.1)
    picked_dirs = {int(np.argmax(np.abs(X[i]))) for i in sel}
    assert len(picked_dirs) == 3               # one of each direction, no duplicate


def test_does_not_saturate_at_low_cosine_rank():
    # a DPP on a rank-limited cosine kernel stops early; D-optimal keeps adding
    # informative points across many directions.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    sel = greedy_fisher_design(X, k=50, ridge=1.0)
    assert len(sel) == 50                       # fills the budget (no rank stall)


def test_logdet_increases_each_step():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 8))
    sel = greedy_fisher_design(X, k=20, ridge=1.0)
    vals = [_logdet(X, sel[:i]) for i in range(1, len(sel) + 1)]
    assert all(b > a - 1e-9 for a, b in zip(vals, vals[1:]))   # monotone non-decreasing


def test_weights_pull_toward_high_fisher_points():
    # two clusters along the same direction: identical geometry, different weight.
    X = np.array([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10)
    w = np.array([0.25] * 10 + [0.01] * 10)     # first cluster far more informative
    sel = greedy_fisher_design(X, weights=w, k=2, ridge=0.01)
    # the very first pick should be from the high-weight cluster
    assert sel[0] < 10


def test_logistic_fisher_weights_peak_at_boundary():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=(n, 1))
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-2.0 * x[:, 0]))).astype(int)
    grid = np.linspace(-4, 4, 41).reshape(-1, 1)
    w, p = logistic_fisher_weights(x, y, grid)
    # Fisher weight w=p(1-p) is largest where p≈0.5, i.e. near x=0
    assert abs(grid[np.argmax(w), 0]) < 1.0
    assert w.max() <= 0.25 + 1e-9
