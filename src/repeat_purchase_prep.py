"""Cleaning, validation folds and final pipelines for the 90-day repeat-purchase project.

Everything here was decided in 02_preprocessing.ipynb; the decision log
(data/preprocessed/decision_log.json) records why.

- Stateless steps (clean_raw and its parts) look at one row at a time, so they
  can be applied to train and test at any moment.
- The pipeline builders return UNFITTED objects. Fit them inside each training
  fold only, so fill values, encoders, scaling and the selection never see the
  rows being scored.
- The saved files still contain free-text columns, the empty customer_type and
  random_noise. column_roles leaves them out, so they never reach a model.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Settings found or decided in the notebooks
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TARGET = "repeat_purchase_90d"
KEYS = ["ID", "customer_id", "purchase_date"]

PLACEHOLDER_DAYS = -4                  # impossible interval value (EDA, D5)
PLACEHOLDER_AMOUNT = 1_000_000_000     # impossible amount filler (EDA, D6)
INTERVAL_COLS = ["days_since_previous_purchase", "prior_mean_gap_days"]
AMOUNT_COLS = ["prior_mean_purchase_value", "spend_365d"]
TOL = 0.01                             # cent rounding tolerance

FREE_TEXT = ["company_name", "address", "street", "postal_code"]   # near-unique, left out (D16)
NOISE = ["random_noise"]                                            # pure noise, left out (D29)
COARSE_DROP = ["city", "postal_area", "municipality"]               # tree branch only (D28)
HISTORY_MARKERS = ["is_first_purchase", "is_single_purchase"]       # linear branch only (D24)
HIGH_CARD_LIMIT = 10                   # more levels -> frequency encoding (D17)
RARE_MIN_ROWS = 20                     # one-hot levels below this are grouped
ENET_C = 0.1                           # elastic-net penalty, a starting point to tune (D30)

GAP_DAYS = 90
HOLDOUT_START = pd.Timestamp("2024-01-01")
DEV_END = HOLDOUT_START - pd.Timedelta(days=GAP_DAYS)   # 2023-10-03
VAL_YEARS = [2020, 2021, 2022, 2023]


# ---------------------------------------------------------------------------
# Stateless cleaning (section 2)
# ---------------------------------------------------------------------------

def repair_frame(df):
    """Row-by-row repairs from the accounting identities (2.4).
    Returns the repaired copy and the rows changed per step."""
    df = df.copy()
    n = {}
    count = df["prior_purchase_count"]
    total = df["prior_total_spend"]

    # Mean purchase value: recompute from total and count
    old_mean = df["prior_mean_purchase_value"]
    new_mean = total / count
    fix_mean = (count > 0) & new_mean.notna() & (
        old_mean.isna() | ((old_mean - new_mean).abs() > TOL)
    )
    df.loc[fix_mean, "prior_mean_purchase_value"] = new_mean[fix_mean]
    n["mean"] = int(fix_mean.sum())

    # One past purchase: mean, min and max all equal the total
    single_cols = ["prior_mean_purchase_value", "prior_min_purchase_value", "prior_max_purchase_value"]
    one = (count == 1) & total.notna()
    off = df[single_cols].sub(total, axis=0).abs().gt(TOL) | df[single_cols].isna()
    fix_single = one & off.any(axis=1)
    for col in single_cols:
        df.loc[one, col] = total[one]
    n["single"] = int(fix_single.sum())

    # Largest purchase above the total is impossible
    bad_max = df["prior_max_purchase_value"] > total + TOL
    df.loc[bad_max, "prior_max_purchase_value"] = np.nan
    n["max"] = int(bad_max.sum())

    # 365-day spend: flip negatives that pass the checks once positive
    s180 = df["spend_180d"]
    s365 = df["spend_365d"]
    flip = (s365 < 0) & (-s365 >= s180 - TOL) & (-s365 <= total + TOL)
    df.loc[flip, "spend_365d"] = -s365[flip]
    n["flip"] = int(flip.sum())

    # 365-day spend: rebuild what is still invalid from the counts
    s365 = df["spend_365d"]
    c365, c180 = df["purchase_count_365d"], df["purchase_count_180d"]
    invalid = (
        s365.isna() | (s365 < 0) | (s365 < s180 - TOL)
        | (s365 > total + TOL) | ((s365 > 0) & (c365 == 0))
    )
    rebuilt = pd.Series(
        np.select([c365 == 0, c365 == c180, c365 == count], [0.0, s180, total], default=np.nan),
        index=df.index,
    )
    n["s365_rebuilt"] = int((invalid & rebuilt.notna()).sum())
    n["s365_blanked"] = int((invalid & s365.notna() & rebuilt.isna()).sum())
    df.loc[invalid, "spend_365d"] = rebuilt[invalid]

    # Mean gap between purchases: rebuild from tenure and days since previous
    tenure = df["customer_tenure_days"]
    days = df["days_since_previous_purchase"]
    gap_value = (tenure - days) / (count - 1)
    fix_gap = (count >= 2) & df["prior_mean_gap_days"].isna() & gap_value.notna() & (gap_value >= 0)
    df.loc[fix_gap, "prior_mean_gap_days"] = gap_value[fix_gap]
    n["gap"] = int(fix_gap.sum())

    # Days since previous purchase: same identity solved the other way; with one purchase it equals tenure
    gap = df["prior_mean_gap_days"]
    days_value = tenure - gap * (count - 1)
    fix_multi = (count >= 2) & days.isna() & days_value.notna() & (days_value >= 0)
    fix_single_days = (count == 1) & days.isna() & tenure.notna()
    days_value = days_value.where(~fix_single_days, tenure)
    fix_days = fix_multi | fix_single_days
    df.loc[fix_days, "days_since_previous_purchase"] = days_value[fix_days]
    n["days"] = int(fix_days.sum())

    return df, n


def add_history_markers(df):
    """First and single-purchase markers (group E)."""
    df = df.copy()
    df["is_first_purchase"] = (df["prior_purchase_count"] == 0).astype(int)
    df["is_single_purchase"] = (df["prior_purchase_count"] == 1).astype(int)
    return df


def clean_raw(df, clientes):
    """All stateless steps of section 2, in order. Works for train and test."""
    df = df.merge(clientes, on="customer_id", how="left", validate="many_to_one")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="%Y-%m-%d")
    df = df.sort_values(["purchase_date", "ID"], kind="stable")
    df = df[df["purchase_date"].notna()].reset_index(drop=True)

    for col in INTERVAL_COLS:
        df[col] = df[col].replace(PLACEHOLDER_DAYS, np.nan)
    for col in AMOUNT_COLS:
        df[col] = df[col].replace(PLACEHOLDER_AMOUNT, np.nan)

    df, _ = repair_frame(df)
    return add_history_markers(df)


# ---------------------------------------------------------------------------
# Validation design (section 3)
# ---------------------------------------------------------------------------

def year_folds(dates, years=VAL_YEARS, gap_days=GAP_DAYS, end=DEV_END):
    """Expanding yearly folds: train on rows whose 90-day window closes before the year starts."""
    dates = pd.Series(dates).reset_index(drop=True)
    folds = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1)
        stop = min(pd.Timestamp(year=year + 1, month=1, day=1), end)
        train_idx = np.flatnonzero(dates < start - pd.Timedelta(days=gap_days))
        val_idx = np.flatnonzero((dates >= start) & (dates < stop))
        folds.append((train_idx, val_idx))
    return folds


def split_dev_holdout(df):
    """Development rows (before DEV_END) and holdout rows (from HOLDOUT_START)."""
    dev = df[df["purchase_date"] < DEV_END].reset_index(drop=True)
    holdout = df[df["purchase_date"] >= HOLDOUT_START].reset_index(drop=True)
    return dev, holdout


def seen_mask(train_part, val_part):
    """True for rows whose customer also appears in the training part."""
    return val_part["customer_id"].isin(train_part["customer_id"]).to_numpy()


def best_threshold(y, p, tol=0.005):
    """Middle of the thresholds whose F1 is within tol of the best (grid 0.05 to 0.95)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    y = np.asarray(y).astype(bool)[:, None]
    pred = np.asarray(p)[:, None] >= grid
    tp = (pred & y).sum(axis=0)
    fp = (pred & ~y).sum(axis=0)
    fn = (~pred & y).sum(axis=0)
    f1s = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    near_top = grid[f1s >= f1s.max() - tol]
    return float(np.median(near_top)), grid, f1s


def f1_floor(y):
    """F1 of predicting 1 for every row."""
    p = np.mean(y)
    return 2 * p / (1 + p)


# ---------------------------------------------------------------------------
# Stateful building blocks (sections 4 and 5)
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each level with its share of training rows. Missing and unseen levels get 0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return pd.DataFrame(
            {f"{c}_freq": X[c].map(self.maps_[c]).astype(float).fillna(0.0) for c in self.columns_},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])


class Log1pColumns(BaseEstimator, TransformerMixin):
    """Apply log1p to the named columns and leave the rest untouched."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_in_).copy()
        cols = [c for c in (self.columns or []) if c in X.columns]
        X[cols] = np.log1p(X[cols].clip(lower=0))  # clip guards against a stray negative in new data
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_in_)


def skewed_columns(df, num, threshold=1.0):
    """Non-negative numeric columns with skewness above threshold (no target involved, as in 5.4)."""
    skip = set(NOISE + ["customer_since_year"])
    candidates = [c for c in num if c not in skip and df[c].min() >= 0]
    skew = df[candidates].skew()
    return skew[skew > threshold].index.tolist()


def column_roles(df):
    """Numeric, large categorical and small categorical inputs, before branch-specific changes."""
    excluded = set(KEYS + [TARGET] + FREE_TEXT + NOISE + HISTORY_MARKERS)
    cols = [
        c for c in df.columns
        if c not in excluded and not c.startswith("flag_") and not df[c].isna().all()
    ]
    text = [c for c in cols if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    num = [c for c in cols if c not in text and pd.api.types.is_numeric_dtype(df[c])]
    levels = df[text].nunique()
    return {
        "num": num,
        "high": levels[levels > HIGH_CARD_LIMIT].index.tolist(),
        "low": levels[levels <= HIGH_CARD_LIMIT].index.tolist(),
    }


# ---------------------------------------------------------------------------
# Final pipelines (sections 5 to 7)
# ---------------------------------------------------------------------------

def _linear_roles(roles):
    return {"num": roles["num"] + HISTORY_MARKERS, "high": roles["high"], "low": roles["low"]}


def _tree_roles(roles):
    keep = lambda cols: [c for c in cols if c not in COARSE_DROP]
    return {"num": keep(roles["num"]), "high": keep(roles["high"]), "low": keep(roles["low"])}


def pipeline_inputs(roles, branch):
    """Original columns a branch's pipeline expects, in order."""
    r = _linear_roles(roles) if branch == "linear" else _tree_roles(roles)
    return r["num"] + r["high"] + r["low"]


def build_linear_pipeline(model, roles, enet_c=ENET_C, log1p_cols=None):
    """Median fill + standard scaling, frequency and one-hot encoding,
    elastic-net selection, then any linear or distance-based model (D19-D30).
    log1p_cols: optional skewed columns to log-transform before the fill (off by default, D21)."""
    r = _linear_roles(roles)
    num_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if log1p_cols:
        num_steps.insert(0, ("log", Log1pColumns(log1p_cols)))
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), r["num"]),
        ("high", Pipeline([
            ("encode", FrequencyEncoder()),
            ("scale", StandardScaler()),
        ]), r["high"]),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=RARE_MIN_ROWS), r["low"]),
    ], sparse_threshold=0)  # always a dense matrix
    selector = SelectFromModel(LogisticRegression(
        l1_ratio=0.5, solver="saga", C=enet_c, max_iter=5000, random_state=RANDOM_STATE,
    ))
    return Pipeline([
        ("prep", prep),
        ("select", selector),
        ("model", clone(model)),
    ])


def build_tree_pipeline(model, roles):
    """Fine geography dropped, small categories as integer codes, large ones as
    frequencies, numbers untouched, then any tree model (D22, D28)."""
    r = _tree_roles(roles)
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), r["low"]),
        ("high", FrequencyEncoder(), r["high"]),
        ("num", "passthrough", r["num"]),
    ], verbose_feature_names_out=False)

    model = clone(model)
    if isinstance(model, HistGradientBoostingClassifier):
        # Small categories come first in the matrix, so their positions are 0..len(low)-1
        model.set_params(categorical_features=list(range(len(r["low"]))))
    return Pipeline([("prep", prep), ("model", model)])
