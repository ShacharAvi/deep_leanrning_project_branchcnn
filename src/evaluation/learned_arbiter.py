"""Learned per-execution arbiter between TAGE-SC-L and a neural model.

The question it answers, for every dynamic execution of a modelled branch:
*trust TAGE or trust the model?* When the two predict the same direction the
choice is irrelevant, so the arbiter is trained only on DISAGREEMENTS, with
target ``1 = the model was right`` (hence TAGE was wrong).

Inputs are what a real predictor has at fetch time:

- the model's logit, its confidence ``|sigmoid - 0.5|`` and predicted direction
- TAGE-SC-L's per-execution internal state (14 columns of
  :data:`src.evaluation.TAGE_FEATURE_GROUPS`: which table answered, confidence,
  provider/alternate disagreement, statistical-corrector sum, its prediction)

Two arbiters per branch:

``learned_lr``   logistic regression -- a handful of multiply-adds, i.e.
                 deployable next to the predictor. The main candidate.
``learned_gbdt`` small gradient-boosted trees -- an "is there more to get"
                 reference, not a hardware proposal.

Honesty rules, same as the threshold arbiter in :mod:`.arbiter`:

1. Fitted on VALIDATION traces only; test arrays only ever reach ``predict``.
2. Its value is judged out-of-fold: validation traces are split into folds
   BY TRACE, each fold's decisions come from a model fitted on the others, and
   the branch keeps the arbiter only if those out-of-fold decisions beat TAGE
   significantly (one-sided exact McNemar, p < alpha). Otherwise the branch
   falls back to TAGE alone.
3. The final arbiter is refitted on all validation traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arbiter import Executions, mcnemar_one_sided


def model_columns(logits: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return np.stack([np.clip(logits, -20, 20) / 10.0,
                     np.abs(p - 0.5) * 2.0,
                     (logits > 0).astype(np.float64)], axis=1)


def tage_columns(log_dir: Path, pc: int, traces: np.ndarray,
                 indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """TAGE state for each (trace, branch_index), exact join or raise."""
    from . import load_prediction_log

    out = None
    names: list[str] = []
    for tid in np.unique(traces):
        sel = traces == tid
        sub = load_prediction_log(Path(log_dir) / str(tid)).for_pc(pc)
        slot = np.searchsorted(sub.branch_index, indices[sel])
        if slot.size and (slot.max() >= len(sub) or not np.array_equal(
                sub.branch_index[slot], indices[sel])):
            raise RuntimeError(f"TAGE log {tid} misses executions of {pc:#x}")
        feats = sub.tage_features()[slot]
        names = sub.tage_feature_names()
        if out is None:
            out = np.zeros((indices.size, feats.shape[1]), dtype=np.float64)
        out[sel] = feats
    return out, names


def arbiter_inputs(ex: Executions, log_dir: Path, pc: int,
                   traces: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tage, _ = tage_columns(log_dir, pc, traces, indices)
    m = model_columns(ex.logits)
    # Do the two disagree is constant on the training rows (all disagree), so
    # it is not an input; what matters is how each one looks when they do.
    return np.concatenate([m, tage], axis=1)


def predictions_differ(ex: Executions) -> np.ndarray:
    """Executions where TAGE and the model predict different directions.

    TAGE's predicted direction is reconstructed from its logged correctness
    and the outcome. That is exactly TAGE's own output -- known at fetch --
    so deciding "do they differ" this way uses no information a real core
    lacks; the arbiter's *choice* on those executions uses only the inputs
    in :func:`arbiter_inputs`.
    """
    tage_pred = np.where(ex.tage_correct, ex.labels > 0, ex.labels <= 0)
    return (ex.logits > 0) != tage_pred


@dataclass
class LearnedArbiter:
    kind: str                 # "lr" or "gbdt"
    model: object | None      # None == never override (fallback to TAGE)
    threshold: float = 0.5
    oof_p: float = 1.0
    oof_gain: int = 0
    trained_on: int = 0

    def use_model(self, ex: Executions, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(ex), dtype=bool)
        differ = predictions_differ(ex)
        use = np.zeros(len(ex), dtype=bool)
        if differ.any():
            prob = self.model.predict_proba(X[differ])[:, 1]
            use[differ] = prob > self.threshold
        return use


def _new(kind: str, seed: int):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1000))
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1,
            l2_regularization=1.0, random_state=seed)
    raise ValueError(kind)


def fit_learned_arbiter(val: Executions, X: np.ndarray, traces: np.ndarray,
                        kind: str = "lr", *, alpha: float = 0.05,
                        folds: int = 3, min_rows: int = 200,
                        seed: int = 0) -> LearnedArbiter:
    """Fit on validation disagreements; keep only if out-of-fold it wins."""
    differ = predictions_differ(val)
    y = val.model_correct[differ].astype(int)
    Xd, groups = X[differ], traces[differ]
    uniq = np.unique(groups)
    if y.size < min_rows or len(np.unique(y)) < 2 or uniq.size < 2:
        return LearnedArbiter(kind, None, trained_on=int(y.size))

    # Out-of-fold decisions, folds by TRACE so no fold sees its own trace.
    rng = np.random.default_rng(seed)
    fold_of = {t: i % min(folds, uniq.size)
               for i, t in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[t] for t in groups])
    oof = np.zeros(y.size, dtype=bool)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        if len(np.unique(y[tr])) < 2:
            continue
        m = _new(kind, seed).fit(Xd[tr], y[tr])
        oof[te] = m.predict_proba(Xd[te])[:, 1] > 0.5
    # On disagreements, using the model is right exactly when y == 1.
    model_only = int((oof & (y == 1)).sum())
    tage_only = int((oof & (y == 0)).sum())
    p = mcnemar_one_sided(model_only, tage_only)
    if model_only - tage_only <= 0 or p >= alpha:
        return LearnedArbiter(kind, None, oof_p=p,
                              oof_gain=model_only - tage_only,
                              trained_on=int(y.size))
    final = _new(kind, seed).fit(Xd, y)
    return LearnedArbiter(kind, final, oof_p=p,
                          oof_gain=model_only - tage_only,
                          trained_on=int(y.size))


def score_mask(ex: Executions, use: np.ndarray) -> dict:
    wrong = np.where(use, ~ex.model_correct, ~ex.tage_correct)
    return {"executions": len(ex), "tage_misses": int((~ex.tage_correct).sum()),
            "hybrid_misses": int(wrong.sum()), "overridden": int(use.sum())}


# ------------------------------------------------------------------ shared

@dataclass
class SharedArbiter:
    """One arbiter for ALL modelled branches of a program.

    Its inputs (model confidence, TAGE's confidence, which table answered,
    ...) mean the same thing on every branch, so pooling the branches'
    validation disagreements gives ~N_branches x more training rows than a
    per-branch arbiter -- and one small arbiter is also cheaper in hardware
    than one per branch. It is branch-agnostic: no branch id is an input.

    ``enabled[name]`` is the per-branch safety gate: a branch uses the shared
    arbiter only if, on ITS validation executions, the shared arbiter's
    out-of-fold decisions beat TAGE significantly. Otherwise that branch
    stays with TAGE.
    """

    kind: str
    model: object | None
    enabled: dict
    oof: dict

    def use_model(self, name: str, ex: Executions, X: np.ndarray) -> np.ndarray:
        use = np.zeros(len(ex), dtype=bool)
        if self.model is None or not self.enabled.get(name):
            return use
        differ = predictions_differ(ex)
        if differ.any():
            use[differ] = self.model.predict_proba(X[differ])[:, 1] > 0.5
        return use


def fit_shared_arbiter(branches: list[tuple[str, Executions, np.ndarray,
                                            np.ndarray]],
                       kind: str = "lr", *, alpha: float = 0.05,
                       folds: int = 3, seed: int = 0) -> SharedArbiter:
    """``branches``: (name, validation Executions, inputs X, trace ids)."""
    rows_X, rows_y, rows_g, rows_b = [], [], [], []
    for name, ex, X, traces in branches:
        differ = predictions_differ(ex)
        rows_X.append(X[differ])
        rows_y.append(ex.model_correct[differ].astype(int))
        rows_g.append(traces[differ])
        rows_b.append(np.full(int(differ.sum()), name))
    X = np.concatenate(rows_X)
    y = np.concatenate(rows_y)
    g = np.concatenate(rows_g)
    b = np.concatenate(rows_b)
    uniq = np.unique(g)
    if y.size < 200 or len(np.unique(y)) < 2 or uniq.size < 2:
        return SharedArbiter(kind, None, {}, {})

    # Out-of-fold by TRACE: a fold's decisions never come from a model that
    # saw that trace, for any branch.
    rng = np.random.default_rng(seed)
    fold_of = {t: i % min(folds, uniq.size)
               for i, t in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[t] for t in g])
    oof = np.zeros(y.size, dtype=bool)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        m = _new(kind, seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1] > 0.5

    enabled, stats = {}, {}
    for name, *_ in branches:
        sel = b == name
        mo = int((oof[sel] & (y[sel] == 1)).sum())
        to = int((oof[sel] & (y[sel] == 0)).sum())
        p = mcnemar_one_sided(mo, to)
        enabled[name] = bool(mo - to > 0 and p < alpha)
        stats[name] = {"oof_gain": mo - to, "oof_p": p, "rows": int(sel.sum())}
    final = _new(kind, seed).fit(X, y)
    return SharedArbiter(kind, final, enabled, stats)


# ------------------------------------------- shared + per-branch correction

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_offset_lr(X: np.ndarray, y: np.ndarray, offset: np.ndarray,
                   lam: float) -> np.ndarray:
    """Logistic regression with a fixed per-row offset (the shared score).

    Minimises  sum logloss(offset + X w + b) + lam/2 ||w||^2 ; the bias is
    not penalised. lam -> inf leaves the shared arbiter unchanged (w = 0);
    lam -> 0 is an unconstrained per-branch arbiter on top of it.
    """
    from scipy.optimize import minimize
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])

    def obj(w):
        z = offset + Xb @ w
        loss = np.sum(np.logaddexp(0.0, z) - y * z)
        reg = 0.5 * lam * np.sum(w[:-1] ** 2)
        grad = Xb.T @ (_sigmoid(z) - y)
        grad[:-1] += lam * w[:-1]
        return loss + reg, grad

    w0 = np.zeros(Xb.shape[1])
    return minimize(obj, w0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 500}).x


@dataclass
class CorrectedSharedArbiter:
    """Shared arbiter + a shrunk per-branch correction (``kind='lr'``)."""

    shared: object | None
    corrections: dict          # name -> (mean, std, w) or None
    lambdas: dict
    enabled: dict
    oof: dict

    def _shared_logit(self, X):
        return self.shared.decision_function(X)

    def use_model(self, name: str, ex: Executions, X: np.ndarray) -> np.ndarray:
        use = np.zeros(len(ex), dtype=bool)
        if self.shared is None or not self.enabled.get(name):
            return use
        differ = predictions_differ(ex)
        if not differ.any():
            return use
        Xd = X[differ]
        z = self._shared_logit(Xd)
        corr = self.corrections.get(name)
        if corr is not None:
            mean, std, w = corr
            Xs = (Xd - mean) / std
            z = z + np.hstack([Xs, np.ones((Xs.shape[0], 1))]) @ w
        use[differ] = z > 0
        return use


LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e6)


def fit_corrected_shared_arbiter(branches, *, alpha: float = 0.05,
                                 folds: int = 3, seed: int = 0,
                                 lambdas=LAMBDAS) -> CorrectedSharedArbiter:
    """``branches``: (name, validation Executions, inputs X, trace ids).

    Honesty: every row's shared score used to fit or judge a correction comes
    from a shared model fitted WITHOUT that row's trace (trace folds), and the
    correction's own out-of-fold decisions use the same folds. Gate as in
    :func:`fit_shared_arbiter`.
    """
    rows = []
    for name, ex, X, traces in branches:
        d = predictions_differ(ex)
        rows.append((name, X[d], ex.model_correct[d].astype(float), traces[d]))
    X = np.concatenate([r[1] for r in rows])
    y = np.concatenate([r[2] for r in rows])
    g = np.concatenate([r[3] for r in rows])
    b = np.concatenate([np.full(r[1].shape[0], r[0]) for r in rows])
    uniq = np.unique(g)
    if y.size < 200 or len(np.unique(y)) < 2 or uniq.size < 2:
        return CorrectedSharedArbiter(None, {}, {}, {}, {})

    rng = np.random.default_rng(seed)
    fold_of = {t: i % min(folds, uniq.size)
               for i, t in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[t] for t in g])
    # Out-of-fold shared logits.
    oof_z = np.zeros(y.size)
    for k in np.unique(fold):
        m = _new("lr", seed).fit(X[fold != k], y[fold != k])
        oof_z[fold == k] = m.decision_function(X[fold == k])
    shared = _new("lr", seed).fit(X, y)

    corrections, lams, enabled, stats = {}, {}, {}, {}
    for name, *_ in branches:
        sel = b == name
        Xb_, yb, zb, fb = X[sel], y[sel], oof_z[sel], fold[sel]
        best = None
        for lam in lambdas:
            # Out-of-fold decisions of shared+correction(lam) on this branch.
            dec = np.zeros(yb.size, dtype=bool)
            for k in np.unique(fb):
                tr, te = fb != k, fb == k
                if tr.sum() < 20 or len(np.unique(yb[tr])) < 2:
                    dec[te] = zb[te] > 0
                    continue
                mean = Xb_[tr].mean(0)
                std = Xb_[tr].std(0) + 1e-9
                w = _fit_offset_lr((Xb_[tr] - mean) / std, yb[tr], zb[tr], lam)
                Xs = (Xb_[te] - mean) / std
                dec[te] = zb[te] + np.hstack(
                    [Xs, np.ones((Xs.shape[0], 1))]) @ w > 0
            gain = int((dec & (yb == 1)).sum()) - int((dec & (yb == 0)).sum())
            if best is None or gain > best[0]:
                best = (gain, lam, dec)
        gain, lam, dec = best
        mo = int((dec & (yb == 1)).sum())
        to = int((dec & (yb == 0)).sum())
        # Bonferroni over the lambda grid, like the threshold arbiter.
        p = mcnemar_one_sided(mo, to)
        enabled[name] = bool(gain > 0 and p < alpha / len(lambdas))
        lams[name] = lam
        stats[name] = {"oof_gain": gain, "oof_p": p, "lambda": lam,
                       "rows": int(sel.sum())}
        if enabled[name] and yb.size >= 20 and len(np.unique(yb)) == 2:
            mean, std = Xb_.mean(0), Xb_.std(0) + 1e-9
            corrections[name] = (mean, std, _fit_offset_lr(
                (Xb_ - mean) / std, yb, zb, lam))
    return CorrectedSharedArbiter(shared, corrections, lams, enabled, stats)


# ------------------------------------------------- extra arbiter inputs

_EXTRA_CACHE: dict = {}


def extra_columns(feature_dir: Path, pc: int, traces: np.ndarray,
                  indices: np.ndarray, groups=("producers", "inst_hist"),
                  ) -> np.ndarray:
    """Per-execution raw feature columns (e.g. dataflow producers) for the
    arbiter, joined exactly on (trace, branch_index); raises on a gap."""
    from ..dataset import load_raw_features
    from .prediction_log import as_unsigned_pc

    out = None
    for tid in np.unique(traces):
        key = (str(feature_dir), str(tid))
        if key not in _EXTRA_CACHE:
            _EXTRA_CACHE[key] = load_raw_features(Path(feature_dir) / f"{tid}.feat")
        bi, pcs, f, side = _EXTRA_CACHE[key]
        keep = np.isin(np.array(side["feature_groups"]), groups)
        mine = pcs == as_unsigned_pc(pc)
        bidx, rows = bi[mine], f[mine][:, keep]
        sel = traces == tid
        slot = np.searchsorted(bidx, indices[sel])
        if slot.size and (slot.max() >= bidx.size or
                          not np.array_equal(bidx[slot], indices[sel])):
            raise RuntimeError(f"extra features {feature_dir}/{tid} miss "
                               f"executions of {pc:#x}")
        if out is None:
            out = np.zeros((indices.size, rows.shape[1]), dtype=np.float64)
        out[sel] = rows[slot]
    return out
