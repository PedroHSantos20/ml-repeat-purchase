"""Scoring and decision tools shared by the modelling notebooks.

Same rules as in 02_preprocessing.ipynb:
- each model is fitted on the training part of every fold and scored on its validation year;
- F1 is computed at the model's own threshold, picked on the four validation years stacked;
- options are compared with a customer-level bootstrap of the F1 difference and a fold-consistency check.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from .repeat_purchase_prep import RANDOM_STATE, TARGET, VAL_YEARS, best_threshold, f1_floor


# ---------------------------------------------------------------------------
# Scoring on the folds
# ---------------------------------------------------------------------------

def count_kept(model):
    """Columns that reach the final model, after encoding and selection."""
    if hasattr(model, "model_"):  # a model that picks its own columns wraps a fitted pipeline
        return count_kept(model.model_)
    steps = model.named_steps
    if "select" in steps:
        sel = steps["select"]
        if hasattr(sel, "n_components_"):  # PCA
            return int(sel.n_components_)
        return int(sel.get_support().sum())
    return len(steps["prep"].get_feature_names_out())


def score_folds(pipe, data, folds, features, target=TARGET, years=VAL_YEARS,
                train_filter=None, train_scores=False):
    """Fit on each fold, return a per-fold table, the stacked predictions and the threshold.

    train_filter: optional function(train_part) -> boolean mask of training rows to keep.
    train_scores: also report log-loss on the training part, to read the overfitting gap.
    """
    rows, oof = [], []
    for (tr, va), year in zip(folds, years):
        train_part, val_part = data.iloc[tr], data.iloc[va]
        if train_filter is not None:
            train_part = train_part[train_filter(train_part)]

        model = clone(pipe).fit(train_part[features], train_part[target])
        p = model.predict_proba(val_part[features])[:, 1]
        y = val_part[target].to_numpy()

        ll = log_loss(y, p)
        ll_floor = log_loss(y, np.full(len(y), train_part[target].mean()))  # always predicting the base rate
        row = {
            "fold": year, "train_rows": len(train_part), "n_features": count_kept(model),
            "log_loss": ll, "ll_floor": ll_floor, "ll_gain": ll_floor - ll,
            "roc_auc": roc_auc_score(y, p),
        }
        if train_scores:
            p_train = model.predict_proba(train_part[features])[:, 1]
            row["train_log_loss"] = log_loss(train_part[target], p_train)
        rows.append(row)

        oof.append(pd.DataFrame({
            "fold": year, "y": y, "p": p,
            "customer_id": val_part["customer_id"].to_numpy(),  # used by the customer bootstrap
        }))

    oof = pd.concat(oof, ignore_index=True)
    thr, _, _ = best_threshold(oof["y"], oof["p"])

    # F1 per fold at this model's own threshold
    for row in rows:
        part = oof[oof["fold"] == row["fold"]]
        row["f1"] = f1_score(part["y"], part["p"] >= thr)
        row["floor"] = f1_floor(part["y"])
        row["lift"] = row["f1"] - row["floor"]

    return pd.DataFrame(rows).set_index("fold"), oof, thr


def summary_row(table, oof, thr):
    """One line per model: fold averages plus F1 on the four years stacked."""
    row = {
        "n_features": table["n_features"].mean(),
        "f1_stacked": f1_score(oof["y"], oof["p"] >= thr),
        "f1_floor": f1_floor(oof["y"]),
        "log_loss": table["log_loss"].mean(),
        "roc_auc": table["roc_auc"].mean(),
        "threshold": thr,
    }
    if "train_log_loss" in table:
        row["train_log_loss"] = table["train_log_loss"].mean()
        row["overfit_gap"] = row["log_loss"] - row["train_log_loss"]  # validation minus training
    return row


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def fast_f1(y, pred):
    """F1 for boolean arrays, quicker than sklearn inside the bootstrap loop."""
    tp = np.sum(y & pred)
    fp = np.sum(~y & pred)
    fn = np.sum(y & ~pred)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap_f1_diff(oof_ref, oof_opt, thr_ref, thr_opt, n_boot=2000, seed=RANDOM_STATE):
    """95% interval of F1(option) - F1(reference), resampling whole customers."""
    assert (oof_ref["customer_id"].to_numpy() == oof_opt["customer_id"].to_numpy()).all()
    y = oof_ref["y"].to_numpy().astype(bool)
    pred_ref = oof_ref["p"].to_numpy() >= thr_ref
    pred_opt = oof_opt["p"].to_numpy() >= thr_opt

    groups = list(oof_ref.groupby("customer_id").indices.values())  # row positions of each customer
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        diffs[b] = fast_f1(y[rows], pred_opt[rows]) - fast_f1(y[rows], pred_ref[rows])
    return np.percentile(diffs, [2.5, 97.5])


def judge(runs, reference, tie_rule=False):
    """Apply the decision rule to scored options.

    runs: name -> (table, oof, threshold), as returned by score_folds.
    tie_rule=True: a tie goes to the option with fewer columns.
    """
    ref_table, ref_oof, ref_thr = runs[reference]
    ref_f1 = fast_f1(ref_oof["y"].to_numpy().astype(bool), ref_oof["p"].to_numpy() >= ref_thr)
    ref_n = ref_table["n_features"].mean()

    rows = []
    for name, (table, oof, thr) in runs.items():
        f1_all = fast_f1(oof["y"].to_numpy().astype(bool), oof["p"].to_numpy() >= thr)
        folds_worse = int((table["f1"] < ref_table["f1"]).sum())
        n = table["n_features"].mean()

        if name == reference:
            lo, hi, verdict = np.nan, np.nan, "reference"
        else:
            lo, hi = bootstrap_f1_diff(ref_oof, oof, ref_thr, thr)
            if lo > 0 and folds_worse <= 1:
                verdict = "better"
            elif hi < 0:
                verdict = "worse"
            elif tie_rule and n < ref_n and folds_worse <= 2:
                verdict = "tie, fewer columns"
            else:
                verdict = "no clear difference"

        rows.append({
            "option": name, "n_features": n, "f1_stacked": f1_all, "f1_diff": f1_all - ref_f1,
            "ci_low": lo, "ci_high": hi, "folds_worse": folds_worse, "threshold": thr,
            "log_loss": table["log_loss"].mean(), "ll_gain": table["ll_gain"].mean(),
            "roc_auc": table["roc_auc"].mean(), "verdict": verdict,
        })
    return pd.DataFrame(rows).set_index("option")
