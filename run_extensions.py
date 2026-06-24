"""
run_extensions.py
=================
Runs independent replication benchmarks alongside modified algorithmic
constraints to observe the sensitivity of the Matthew's Correlation Coefficient (MCC)
and False Positive Rate (FPR).
"""

import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from independent_replicate import load, split, clean

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, confusion_matrix

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "results" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def execute_scenario(data_tuple, max_iter, balance_factor, scenario_name):
    data_linux, data_pass, nft121, nft123 = data_tuple
    train_builds, test_builds = split(data_linux)

    train_linux = data_linux[data_linux["buildId"].isin(train_builds)].sample(n=100_000, random_state=42)
    test_linux = data_linux[data_linux["buildId"].isin(test_builds)].sample(n=30_000, random_state=42)

    train_df = clean(pd.concat([
        train_linux[train_linux["label"] == 0].assign(target=1),
        train_linux[train_linux["label"] == 1].assign(target=0),
        pd.concat([
            data_pass[(data_pass["buildId"] == 121_238) & (data_pass["testId"].isin(nft121))],
            data_pass[(data_pass["buildId"] == 123_038) & (data_pass["testId"].isin(nft123))]
        ]).assign(target=0)
    ], ignore_index=True))

    test_df = clean(test_linux[test_linux["label"].isin([0, 1])])
    test_df["target"] = (test_df["label"] == 0).astype(int)

    pos_count = (train_df["target"] == 1).sum()
    neg_count = (train_df["target"] == 0).sum()
    base_weight = neg_count / pos_count if pos_count > 0 else 1.0
    adjusted_weight = base_weight * balance_factor

    weights = {1: adjusted_weight, 0: 1.0}

    feature_processor = ColumnTransformer([
        ("text_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=100), "testSource"),
        ("suite_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["testSuite"]),
        ("numerical_scale", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler())
        ]), ["runDuration", "flakeRate"])
    ], remainder="drop")

    pipeline = Pipeline([
        ("features", feature_processor),
        ("boosting_engine", HistGradientBoostingClassifier(
            max_iter=max_iter,
            random_state=42
        ))
    ])

    x_train = train_df
    y_train = train_df["target"].values

    sample_weights = np.array([weights[t] for t in y_train])

    pipeline.fit(x_train, y_train, boosting_engine__sample_weight=sample_weights)
    predictions = pipeline.predict(test_df)
    y_true = test_df["target"].values

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "Scenario": scenario_name,
        "Max_Iter": max_iter,
        "Balance_Factor": balance_factor,
        "Precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "Recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, predictions), 4),
        "FPR": round(fpr, 4),
        "Raw_Log": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    }


def main():
    print("=" * 80)
    print("EXTENSION MATRIX EXPERIMENTS & ENSEMBLE PERTURBATIONS")
    print("=" * 80)

    data_tuple = load()

    iterations = [50, 100, 200]
    balance_factors = [1.0, 1.5, 2.0]

    records = []

    print("\nExecuting Baseline Replication Run...")
    base_record = execute_scenario(data_tuple, max_iter=100, balance_factor=1.0, scenario_name="Clean-Room Baseline")
    records.append(base_record)
    print(f" -> Baseline Complete: MCC={base_record['MCC']}, FPR={base_record['FPR']}")

    print("\nLaunching Extension Matrix Parameter Perturbations...")
    for idx, (mit, bfac) in enumerate(itertools.product(iterations, balance_factors), 1):
        if mit == 100 and bfac == 1.0:
            continue
        s_name = f"Extension_Variant_{idx}"
        print(f" -> Testing {s_name}: Max_Iter={mit}, Weight_Skew={bfac}...")
        rec = execute_scenario(data_tuple, max_iter=mit, balance_factor=bfac, scenario_name=s_name)
        records.append(rec)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(RESULTS_DIR / "extension_runs_matrix.csv", index=False)

    with open(LOGS_DIR / "task7_execution_record.log", "w") as f:
        f.write("=== TASK 7 RAW LOG EXECUTION RECORD ===\n")
        f.write(summary_df.to_string(index=False))

    print("\n" + "-" * 40 + "\nEXTENSION MATRIX PROCESS RUNS COMPLETE\n" + "-" * 40)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
