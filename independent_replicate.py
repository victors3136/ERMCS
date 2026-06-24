"""
independent_replicate.py
========================
Clean-room independent replication script for Task 6.
Implements the alternative architecture designed in Task 5.

Key Design Divergences from Authors' Pipeline:
1. Text Engine: Character N-Grams (range 3-5) TF-IDF instead of simple Bag-of-Words CountVectorizer.
2. Learning Engine: Histogram-Based Gradient Boosting with native cost-sensitive class balancing.
3. No synthetic generation (SMOTE skipped) to protect text feature space sparsity.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load():
    if not (DATA_DIR / "dataset.35past.Linux10k.json").exists():
        raise FileNotFoundError(f"Missing baseline telemetry files in {DATA_DIR}")

    data_linux = pd.read_json(DATA_DIR / "dataset.35past.Linux10k.json")
    data_pass = pd.read_json(DATA_DIR / "dataset.pass.json")

    with open(DATA_DIR / "nft-121.json") as f:
        nft121 = set(json.load(f))
    with open(DATA_DIR / "nft-123.json") as f:
        nft123 = set(json.load(f))

    return data_linux, data_pass, nft121, nft123

TRAIN_SPLIT = 0.8

def split(data, train_pct=TRAIN_SPLIT):
    unique_builds = np.sort(data["buildId"].unique())
    split_idx = int(len(unique_builds) * train_pct)
    return set(unique_builds[:split_idx]), set(unique_builds[split_idx:])

def clean(data):
    return data[data["testSource"].astype(str).str.strip() != ""].copy()


def rq3(data_linux, data_pass, nft121, nft123):
    print("\n" + "=" * 80)
    print("RUNNING HIST-GRADIENT BOOSTING PIPELINE")
    print("=" * 80)

    train_builds, test_builds = split(data_linux)

    train_linux = data_linux[data_linux["buildId"].isin(train_builds)].sample(n=150_000, random_state=42)
    test_linux = data_linux[data_linux["buildId"].isin(test_builds)].sample(n=40_000, random_state=42)

    flaky_train = train_linux[train_linux["label"] == 0]
    fault_train = train_linux[train_linux["label"] == 1]

    pass121 = data_pass[(data_pass["buildId"] == 121_238) & (data_pass["testId"].isin(nft121))]
    pass123 = data_pass[(data_pass["buildId"] == 123_038) & (data_pass["testId"].isin(nft123))]
    pass_train = pd.concat([pass121, pass123])

    raw_train = pd.concat([
        flaky_train.assign(target=1),
        fault_train.assign(target=0),
        pass_train.assign(target=0)
    ], ignore_index=True)

    train_df = clean(raw_train)
    test_df = clean(test_linux[test_linux["label"].isin([0, 1])])
    test_df["target"] = (test_df["label"] == 0).astype(int)

    print(f" -> Clean Train Matrix Size: {train_df.shape[0]} rows")
    print(f" -> Clean Evaluation Deck Size: {test_df.shape[0]} rows")

    feature_processor = ColumnTransformer([
        ("text_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=150), "testSource"),
        ("suite_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["testSuite"]),
        ("numerical_scale", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler())
        ]), ["runDuration", "flakeRate"])
    ], remainder="drop")

    pipeline = Pipeline([
        ("features", feature_processor),
        ("boosting_engine", HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            random_state=42
        ))
    ])

    print("\nFitting Class-Weighted Gradient Boosting Framework...")
    pipeline.fit(train_df, train_df["target"])

    print("Executing Inference across Evaluation Holdout Deck...")
    predictions = pipeline.predict(test_df)
    y_true = test_df["target"].values

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    mcc = matthews_corrcoef(y_true, predictions)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "Pipeline_Type": "Clean-Room GradientBoosting (TF-IDF)",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "MCC": f"{mcc:.4f}",
        "FPR (Missed Faults Rate)": f"{fpr:.4f}",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

    results_df = pd.DataFrame([metrics])
    results_df.to_csv(RESULTS_DIR / "independent_results.csv", index=False)

    print("\n" + "-" * 40 + "\nRESULTS SUMMARY\n" + "-" * 40)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return metrics


if __name__ == "__main__":
    try:
        linux_df, pass_df, n121, n123 = load()
        rq3(linux_df, pass_df, n121, n123)
    except Exception as e:
        print(f"\n[Execution Error]: {str(e)}")