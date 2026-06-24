"""
reproduce.py
============
Reproduction pipeline for:

    Haben, G., Habchi, S., Papadakis, M., Cordy, M., Le Traon, Y. (2024).
    "The Importance of Discerning Flaky from Fault-triggering Test Failures:
    A Case Study on the Chromium CI." ASE 2024 (CORE A*).
    Replication package: https://github.com/GuillaumeHaben/ChromiumFlakyFailures

This script re-implements the methodology described in Section 5.2 of the
paper (not a line-for-line copy of the authors' notebook, which used
deprecated pandas/sklearn APIs from 2022-2023). Every modelling choice below
is annotated with the paper section/table it comes from so a reader can
verify the mapping paper -> code.

WHAT EACH RQ ANSWERS (paper Section 5.1)
-----------------------------------------
RQ1: Train a vocabulary-based model on TESTS (flaky vs non-flaky), evaluate
     on FAILURES in held-out builds. -> Table 4.
RQ2: Train the same vocabulary-based approach on FAILURES instead of tests.
     -> Table 5, row "No" (execution features).
RQ3: Same as RQ2 but add dynamic execution features (flakeRate, runDuration).
     -> Table 5, row "Yes" (execution features).

DATA SPLIT (paper Section 5.2.4)
---------------------------------
Time-sensitive split: first 80% of builds (chronological) = train,
last 20% = holdout/test. This is NOT a random split -- random splitting
inflates performance because it leaks future information into training
(the paper explicitly warns about this).

ASSUMPTIONS YOU MUST VERIFY AGAINST THE REAL DATA (run inspect_data.py first)
------------------------------------------------------------------------------
1. Label encoding on dataset.35past.Linux10k.json: 0=flaky, 1=fault-revealing
   ("legit"), confirmed by the authors' notebook comment. dataset.pass.json
   uses label=2 for passing test executions.
2. The paper trains RQ1 on tests as of "the 8,000th build" for the passing
   class. The replication package ships passing-test snapshots for TWO
   build numbers (the files nft-121.json / nft-123.json correspond to two
   build IDs, e.g. 121238 and 123038). Which one represents the paper's
   "b_8000" boundary is not self-evident from the README alone -- this
   script defaults to using the UNION of both (clearly flagged below) and
   exposes PASS_BUILD_STRATEGY so you can switch to a single one and compare.
   Document whichever choice you make -- this is a legitimate
   "replication package ambiguity" finding for your report.
3. "Clean" non-flaky tests are restricted to those listed in nft-121 / nft-123
   (Never-found-to-be-Flaky-or-faulTy), per the README's variable naming.

Usage
-----
    python reproduce.py --rq all
    python reproduce.py --rq 1
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# --- CONFIG: tunable assumptions, see docstring point 2 above -------------
TRAIN_FRACTION = 0.8          # paper Sec 5.2.4: first 80% builds = train
MAX_VOCAB_FEATURES = 100      # matches authors' notebook: CountVectorizer(max_features=100)
RANDOM_STATE = 42
PASS_BUILD_STRATEGY = "union"  # "union" or a specific build id, see assumption #2


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_raw_data():
    """Load the four files exactly as named in the replication package README."""
    data_linux = pd.read_json(DATA_DIR / "dataset.35past.Linux10k.json")
    data_pass = pd.read_json(DATA_DIR / "dataset.pass.json")
    with open(DATA_DIR / "nft-121.json") as f:
        nft121 = json.load(f)
    with open(DATA_DIR / "nft-123.json") as f:
        nft123 = json.load(f)
    return data_linux, data_pass, nft121, nft123


def chronological_build_split(build_ids, train_fraction=TRAIN_FRACTION):
    """Sort unique build IDs chronologically and split first 80% / last 20%.

    Paper Sec 5.2.4: 'the first 80% builds are selected as a training set
    and the last 20% as a holdout set' -- this respects time ordering and
    avoids data leakage from random splitting.
    """
    unique_builds = np.sort(build_ids.unique())
    cutoff_idx = int(len(unique_builds) * train_fraction)
    train_builds = set(unique_builds[:cutoff_idx])
    test_builds = set(unique_builds[cutoff_idx:])
    return train_builds, test_builds


def first_nonempty_source(df, source_col="testSource", id_col="testId"):
    """Collapse failure-level rows to one row per unique test, keeping the
    first non-empty testSource found for that test (used for RQ1, which
    is trained on TESTS, not individual failures -- paper Sec 5.2.2)."""
    df = df[df[source_col].astype(str).str.strip() != ""]
    return df.drop_duplicates(subset=[id_col], keep="first")


# ---------------------------------------------------------------------------
# MODEL PIPELINES (paper Sec 5.2.5)
# ---------------------------------------------------------------------------

def build_vocab_pipeline():
    """Vocabulary-only pipeline used for RQ1 and RQ2.

    CountVectorizer (bag-of-words, max_features=100) -> SelectKBest(chi2)
    -> SMOTE -> BalancedRandomForestClassifier. Matches paper Sec 5.2.5:
    bag-of-words representation, feature selection via chi2, balanced
    random forest to cope with severe class imbalance (the minority class
    is ~1% of the data).
    """
    preprocessor = ColumnTransformer(
        [("testSource", CountVectorizer(max_features=MAX_VOCAB_FEATURES), "testSource")],
        remainder="drop",
    )
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(chi2, k="all")),  # k tuned via grid search below
        # ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return pipe


def build_extended_pipeline():
    """Vocabulary + dynamic execution features, used for RQ3.

    Adds flakeRate (binned, paper Sec 4.3) and runDuration (scaled) on top
    of the vocabulary features, per Table 2 and Sec 6.3 of the paper.
    """
    preprocessor = ColumnTransformer([
        ("testSource", CountVectorizer(max_features=MAX_VOCAB_FEATURES), "testSource"),
        ("testSuite", OneHotEncoder(handle_unknown="ignore"), ["testSuite"]),
        ("flakeRate", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("bin", KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")),
        ]), ["flakeRate"]),
        ("runDuration", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),  # chi2 requires non-negative features
        ]), ["runDuration"]),
    ], remainder="drop")
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(chi2, k="all")),
        # ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return pipe


GRID = {
    "clf__n_estimators": [100],
    "select__k": ["all"],
}


def fit_with_grid_search(pipe, X_train, y_train, cv=3):
    """Paper Sec 5.2.5: 'Hyperparameters ... are tuned using a grid search
    approach and cross-validation in the training set. Once optimized, we
    retrain a model fitted on the whole training set.'
    """
    search = GridSearchCV(pipe, GRID, scoring="matthews_corrcoef", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# METRICS (paper Sec 5.2.6)
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, label, save_confmat_path=None):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    metrics = {
        "label": label,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "fpr": fpr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": len(y_test),
    }

    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        if k != "label":
            print(f"  {k}: {v}")

    if save_confmat_path:
        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred, labels=[0, 1]),
            display_labels=["Non-Flaky", "Flaky"],
        )
        disp.plot(cmap="Blues", values_format=".0f")
        disp.figure_.savefig(save_confmat_path, bbox_inches="tight")
        disp.figure_.clf()

    return metrics


# ---------------------------------------------------------------------------
# RQ1: train on TESTS, evaluate on FAILURES (Table 4)
# ---------------------------------------------------------------------------

def run_rq1(data_linux, data_pass, nft121, nft123):
    print("\n" + "=" * 70 + "\nRQ1: vocabulary model trained on TESTS\n" + "=" * 70)

    train_builds, test_builds = chronological_build_split(data_linux["buildId"])

    train_linux = data_linux[data_linux["buildId"].isin(train_builds)].sample(
        n=200000,
        random_state=42
    )
    flaky_tests = first_nonempty_source(train_linux[train_linux["label"] == 0])
    legit_tests = first_nonempty_source(train_linux[train_linux["label"] == 1])

    # Passing tests: filter to the "clean" never-found-to-be-flaky-or-faulty
    # set, per assumption #2/#3 in the module docstring.
    pass121 = data_pass[(data_pass["buildId"] == 121238) & (data_pass["testId"].isin(nft121))]
    pass123 = data_pass[(data_pass["buildId"] == 123038) & (data_pass["testId"].isin(nft123))]
    passing_tests = first_nonempty_source(pd.concat([pass121, pass123]))

    # Remove any passing/legit test that ever appeared as flaky (paper:
    # "minus the tests that are found as flaky in any of the builds under study")
    flaky_ids = set(flaky_tests["testId"])
    legit_tests = legit_tests[~legit_tests["testId"].isin(flaky_ids)]
    passing_tests = passing_tests[~passing_tests["testId"].isin(flaky_ids)]

    print(f"  Train set -> flaky tests: {len(flaky_tests)}, "
          f"fault-revealing tests: {len(legit_tests)}, "
          f"passing tests: {len(passing_tests)}")
    print("  (paper reports 8,857 / 910 / 69,159 respectively -- compare here)")

    train_df = pd.concat([
        flaky_tests.assign(y=1),
        legit_tests.assign(y=0),
        passing_tests.assign(y=0),
    ], ignore_index=True)

    # Evaluation: individual FAILURES (not deduplicated) in the held-out builds.
    test_linux = data_linux[data_linux["buildId"].isin(test_builds)].sample(
        n=50000,
        random_state=42
    )
    test_df = test_linux[test_linux["label"].isin([0, 1])].copy()
    test_df["y"] = (test_df["label"] == 0).astype(int)  # 1 = flaky failure

    print(f"  Test set -> flaky failures: {(test_df['y'] == 1).sum()}, "
          f"fault-triggering failures: {(test_df['y'] == 0).sum()}")
    print("  (paper reports 217,503 / 2,320 respectively -- compare here)")

    pipe = build_vocab_pipeline()
    model = fit_with_grid_search(pipe, train_df, train_df["y"])
    return evaluate(model, test_df, test_df["y"], "RQ1",
                     FIGURES_DIR / "rq1_confusion_matrix.png")


# ---------------------------------------------------------------------------
# RQ2: train on FAILURES, vocabulary only (Table 5, row "No")
# ---------------------------------------------------------------------------

def build_rq2_rq3_sets(data_linux, data_pass, nft121, nft123):
    """Shared train/test construction for RQ2 and RQ3.

    Paper Sec 5.2.3: 'we train our classifier on non-flaky executions
    (passing and fault-revealing tests execution) and flaky failures.'
    Test set is the same flaky-vs-fault-triggering FAILURES used in RQ1
    ('The test set is common in all RQs.').
    """
    train_builds, test_builds = chronological_build_split(data_linux["buildId"])
    train_linux = data_linux[data_linux["buildId"].isin(train_builds)].sample(
        n=200000,
        random_state=42
    )

    flaky_failures = train_linux[train_linux["label"] == 0].copy()
    legit_executions = train_linux[train_linux["label"] == 1].copy()

    pass121 = data_pass[(data_pass["buildId"] == 121238) & (data_pass["testId"].isin(nft121))]
    pass123 = data_pass[(data_pass["buildId"] == 123038) & (data_pass["testId"].isin(nft123))]
    passing_executions = pd.concat([pass121, pass123])

    train_df = pd.concat([
        flaky_failures.assign(y=1),
        legit_executions.assign(y=0),
        passing_executions.assign(y=0),
    ], ignore_index=True)
    train_df = train_df[train_df["testSource"].astype(str).str.strip() != ""]

    test_linux = data_linux[data_linux["buildId"].isin(test_builds)].sample(
        n=50000,
        random_state=42
    )
    test_df = test_linux[test_linux["label"].isin([0, 1])].copy()
    test_df["y"] = (test_df["label"] == 0).astype(int)

    return train_df, test_df


def run_rq2(data_linux, data_pass, nft121, nft123):
    print("\n" + "=" * 70 + "\nRQ2: vocabulary model trained on FAILURES\n" + "=" * 70)
    train_df, test_df = build_rq2_rq3_sets(data_linux, data_pass, nft121, nft123)
    pipe = build_vocab_pipeline()
    model = fit_with_grid_search(pipe, train_df, train_df["y"])
    return evaluate(model, test_df, test_df["y"], "RQ2",
                     FIGURES_DIR / "rq2_confusion_matrix.png")


# ---------------------------------------------------------------------------
# RQ3: train on FAILURES + dynamic features (Table 5, row "Yes")
# ---------------------------------------------------------------------------

def run_rq3(data_linux, data_pass, nft121, nft123):
    print("\n" + "=" * 70 + "\nRQ3: vocabulary + execution features\n" + "=" * 70)
    train_df, test_df = build_rq2_rq3_sets(data_linux, data_pass, nft121, nft123)
    pipe = build_extended_pipeline()
    model = fit_with_grid_search(pipe, train_df, train_df["y"])
    return evaluate(model, test_df, test_df["y"], "RQ3",
                     FIGURES_DIR / "rq3_confusion_matrix.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    data_linux, data_pass, nft121, nft123 = load_raw_data()
    print(f"Loaded dataLinux: {data_linux.shape}, dataPass: {data_pass.shape}")

    results = []
    if args.rq in ("1", "all"):
        results.append(run_rq1(data_linux, data_pass, nft121, nft123))
    if args.rq in ("2", "all"):
        results.append(run_rq2(data_linux, data_pass, nft121, nft123))
    if args.rq in ("3", "all"):
        results.append(run_rq3(data_linux, data_pass, nft121, nft123))

    out_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "reproduced_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved reproduced results -> {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
