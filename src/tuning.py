"""Hyperparameter search with Optuna on cached fold matrices.

Week 5 idea: when only the model's settings change, the preprocessing is the same
for every candidate, so it is fitted once per fold and its output reused.

A search is defined by three functions written in the notebook:
- suggest(trial)    -> dict of settings, using only trial.suggest_* calls;
- get_cache(params) -> the FoldCache those settings need (e.g. with or without log1p);
- make_model(params)-> an unfitted estimator that runs on the cached matrices.
"""

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from .repeat_purchase_prep import RANDOM_STATE, TARGET, VAL_YEARS, best_threshold, f1_floor

OBJECTIVES = {"log_loss": "minimize", "f1": "maximize"}


class FoldCache:
    """Preprocessing fitted once per fold; training and validation matrices kept for every trial."""

    def __init__(self, prep, data, folds, features, target=TARGET, years=VAL_YEARS):
        self.folds = []
        for (tr, va), year in zip(folds, years):
            train_part, val_part = data.iloc[tr], data.iloc[va]
            fitted = clone(prep).fit(train_part[features], train_part[target])
            self.folds.append({
                "year": year,
                "X_train": np.asarray(fitted.transform(train_part[features]), dtype=float),
                "y_train": train_part[target].to_numpy(),
                "X_val": np.asarray(fitted.transform(val_part[features]), dtype=float),
                "y_val": val_part[target].to_numpy(),
                "customer_id": val_part["customer_id"].to_numpy(),  # for the customer bootstrap
            })


def _columns_used(model, X):
    """Columns the model actually uses: after selection if there is a selector."""
    if hasattr(model, "named_steps") and "select" in model.named_steps:
        return int(model.named_steps["select"].get_support().sum())
    return X.shape[1]


def evaluate_cached(cache, model):
    """Fit the model on each cached fold. Same output as evaluation.score_folds: (table, oof, threshold)."""
    rows, oof = [], []
    for f in cache.folds:
        fitted = clone(model).fit(f["X_train"], f["y_train"])
        p = fitted.predict_proba(f["X_val"])[:, 1]
        p_train = fitted.predict_proba(f["X_train"])[:, 1]

        ll = log_loss(f["y_val"], p)
        ll_floor = log_loss(f["y_val"], np.full(len(p), f["y_train"].mean()))
        rows.append({
            "fold": f["year"], "train_rows": len(f["y_train"]), "n_features": _columns_used(fitted, f["X_train"]),
            "log_loss": ll, "ll_floor": ll_floor, "ll_gain": ll_floor - ll,
            "roc_auc": roc_auc_score(f["y_val"], p), "train_log_loss": log_loss(f["y_train"], p_train),
        })
        oof.append(pd.DataFrame({"fold": f["year"], "y": f["y_val"], "p": p, "customer_id": f["customer_id"]}))

    oof = pd.concat(oof, ignore_index=True)
    thr, _, _ = best_threshold(oof["y"], oof["p"])
    for row in rows:
        part = oof[oof["fold"] == row["fold"]]
        row["f1"] = f1_score(part["y"], part["p"] >= thr)
        row["floor"] = f1_floor(part["y"])
        row["lift"] = row["f1"] - row["floor"]
    return pd.DataFrame(rows).set_index("fold"), oof, thr


def tune(get_cache, make_model, suggest, objective="log_loss", n_trials=60, seed=RANDOM_STATE, name=None,
         show_progress_bar=True, storage=None, fresh=False):
    """Run an Optuna search steered by `objective` ("log_loss" or "f1").
    Every trial records both metrics, plus ROC-AUC, threshold, training log-loss and columns used.

    show_progress_bar: tqdm bar with the best value so far (needs ipywidgets in a notebook).
    storage: e.g. "sqlite:///path/studies.db". Finished trials are saved there; a rerun with the same
        name reloads them and only runs what is missing to reach n_trials.
    fresh: delete a stored study with this name first. Use it, or a new name, whenever the
        search itself changes (ranges, settings, data), so old trials are not reused by mistake.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}, got {objective!r}")
    if storage is not None and name is None:
        raise ValueError("A stored study needs a name, e.g. 'logistic_log_loss'")
    if storage is not None and fresh:
        try:
            optuna.delete_study(study_name=name, storage=storage)
        except KeyError:
            pass  # nothing stored under this name yet

    def run(trial):
        params = suggest(trial)
        table, oof, thr = evaluate_cached(get_cache(params), make_model(params))
        f1 = f1_score(oof["y"], oof["p"] >= thr)
        ll = table["log_loss"].mean()
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("log_loss", ll)
        trial.set_user_attr("roc_auc", table["roc_auc"].mean())
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("train_log_loss", table["train_log_loss"].mean())
        trial.set_user_attr("overfit_gap", ll - table["train_log_loss"].mean())
        trial.set_user_attr("n_features", table["n_features"].mean())
        return ll if objective == "log_loss" else f1

    study = optuna.create_study(
        direction=OBJECTIVES[objective],
        sampler=optuna.samplers.TPESampler(seed=seed),  # same seed -> same trials on a rerun
        study_name=name or objective,
        storage=storage,
        load_if_exists=storage is not None,
    )
    if study.direction.name.lower() != OBJECTIVES[objective]:
        raise ValueError(f"Stored study {name!r} was steered the other way; use a new name or fresh=True")
    study.set_user_attr("objective", objective)  # lets the helpers below sort by the steering metric

    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < n_trials:
        study.optimize(run, n_trials=n_trials - done, show_progress_bar=show_progress_bar)
    return study


def chosen_params(chosen, studies, separator="_trial_"):
    """Settings behind a chosen name like 'f1_trial_32'; empty for 'base' (the untuned pipeline)."""
    if chosen == "base":
        return {}
    search, number = chosen.rsplit(separator, 1)
    return dict(studies[search].trials[int(number)].params)


def trials_table(study):
    """All finished trials with their settings and every recorded metric, best first."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    df = pd.DataFrame([{"trial": t.number, **t.params, **t.user_attrs} for t in done]).set_index("trial")
    key = study.user_attrs["objective"]
    return df.sort_values(key, ascending=(OBJECTIVES[key] == "minimize"))


def top_params(study, k=5):
    """Settings of the k best trials, as {trial_number: params}.
    Read from the trials themselves, so integers stay integers (a table row would turn them into floats)."""
    order = trials_table(study).head(k).index
    by_number = {t.number: t.params for t in study.trials}
    return {int(n): dict(by_number[n]) for n in order}


def rescore(params_by_trial, get_cache, make_model, prefix="trial_"):
    """Re-run chosen settings and return runs in the format evaluation.judge expects."""
    runs = {}
    for number, params in params_by_trial.items():
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}  # numpy -> python
        runs[f"{prefix}{number}"] = evaluate_cached(get_cache(params), make_model(params))
    return runs


def plot_trials(studies, title=""):
    """Left: every trial's log-loss against its F1. Right: validation against training log-loss.
    studies: {label: study}, so searches steered by different metrics share one chart."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))
    for (label, study), marker in zip(studies.items(), ["o", "^", "s", "D"]):
        df = trials_table(study)
        ax_left.scatter(df["log_loss"], df["f1"], alpha=0.6, marker=marker, label=label)
        ax_right.scatter(df["train_log_loss"], df["log_loss"], alpha=0.6, marker=marker, label=label)

    ax_left.set_xlabel("Validation log-loss (lower is better)")
    ax_left.set_ylabel("Stacked F1 at own threshold")
    ax_left.set_title("Log-loss against F1, every trial")

    lims = [min(ax_right.get_xlim()[0], ax_right.get_ylim()[0]), max(ax_right.get_xlim()[1], ax_right.get_ylim()[1])]
    ax_right.plot(lims, lims, ls="--", color="grey", lw=1, label="no overfitting")
    ax_right.set_xlabel("Training log-loss")
    ax_right.set_ylabel("Validation log-loss")
    ax_right.set_title("Overfitting: distance above the line")

    for ax in (ax_left, ax_right):
        ax.legend(frameon=False)
    fig.suptitle(title)
    plt.tight_layout()
    return fig
