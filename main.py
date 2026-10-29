"""Current-condition failure triage on public synthetic AI4I measurements."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
SOURCE_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
FEATURES = NUMERIC + ["Type"]
TARGET = "Machine failure"
FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SEED = 42


def load_data(path=None):
    path = Path(path) if path is not None else ROOT / "data" / "ai4i2020.csv"
    if not path.exists():
        with urlopen(SOURCE_URL, timeout=30) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                raw = archive.read("ai4i2020.csv")
        if sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded source differs from the audited version.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source checksum mismatch; restore the unmodified UCI CSV.")
    data = pd.read_csv(path)
    if data.shape != (10000, 14) or data.isna().any().any():
        raise ValueError("Expected 10,000 complete source rows and 14 columns.")
    if not np.array_equal(data["UDI"], np.arange(1, len(data) + 1)):
        raise ValueError("Unexpected source row sequence.")
    if not data["Type"].isin(["L", "M", "H"]).all() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Unexpected type or target codes.")
    if not np.isfinite(data[NUMERIC].to_numpy()).all():
        raise ValueError("Measurements must be finite.")
    flag_union = data[FAILURE_FLAGS].max(axis=1)
    audit = {"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256, "source_is_synthetic": True,
             "rows": len(data), "missing_values": int(data.isna().sum().sum()),
             "duplicate_udi": int(data["UDI"].duplicated().sum()),
             "duplicate_product_id": int(data["Product ID"].duplicated().sum()),
             "type_counts": data["Type"].value_counts().to_dict(),
             "target_counts": data[TARGET].value_counts().sort_index().to_dict(),
             "flag_union_disagreements": int((flag_union != data[TARGET]).sum()),
             "failure_label_without_flag": int(((data[TARGET] == 1) & (flag_union == 0)).sum()),
             "flag_without_failure_label": int(((data[TARGET] == 0) & (flag_union == 1)).sum()),
             "label_handling": "Retain published Machine failure labels unchanged; no relabeling from auxiliary flags."}
    return data, audit


def split_source_order(data, train_fraction=0.8):
    cutoff = int(len(data) * train_fraction)
    if not 0 < cutoff < len(data):
        raise ValueError("Need nonempty training and test partitions.")
    train, test = data.iloc[:cutoff].copy(), data.iloc[cutoff:].copy()
    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError("Both partitions need failure and non-failure observations for this evaluation.")
    return train, test


def build_tree():
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("type", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["Type"]),
    ])
    # Fixed tutorial settings, not selected by searching the test results.
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=SEED)
    return Pipeline([("preprocessing", preprocessing), ("tree", tree)])


def binary_metrics(actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "failure_precision": float(precision), "failure_recall": float(recall), "failure_f1": float(f1),
            "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}


def evaluate(train, test):
    model = build_tree().fit(train[FEATURES], train[TARGET])
    baseline = DummyClassifier(strategy="most_frequent").fit(train[FEATURES], train[TARGET])
    predictions = test[["UDI", "Product ID"] + FEATURES + [TARGET]].copy()
    predictions["majority_baseline"] = baseline.predict(test[FEATURES])
    predictions["shallow_tree"] = model.predict(test[FEATURES])
    positive_column = list(model.named_steps["tree"].classes_).index(1)
    # Class weighting changes these leaf fractions; do not call them calibrated risks.
    predictions["weighted_tree_score"] = model.predict_proba(test[FEATURES])[:, positive_column]
    comparison = pd.DataFrame([{"method": name, **binary_metrics(test[TARGET], predictions[name])}
                               for name in ["majority_baseline", "shallow_tree"]])
    return model, predictions, comparison


def save_charts(train, predictions, comparison, model, feature_names, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for name, axis in zip(comparison["method"], axes):
        matrix = confusion_matrix(predictions[TARGET], predictions[name], labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["No failure", "Failure"]).plot(
            ax=axis, colorbar=False, values_format="d", cmap="Blues")
        axis.set_title(name)
    fig.suptitle("Synthetic source-order holdout: current failure labels", fontweight="bold")
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    pairs = [("Torque [Nm]", "Rotational speed [rpm]"), ("Air temperature [K]", "Process temperature [K]")]
    for axis, (x, y) in zip(axes, pairs):
        for label, color, alpha in [(0, "#007F78", 0.15), (1, "#D97732", 0.8)]:
            group = train[train[TARGET] == label]
            axis.scatter(group[x], group[y], s=8 if label == 0 else 18, alpha=alpha, color=color,
                         label=f"{'Failure' if label else 'No failure'} (n={len(group)})")
        axis.set(xlabel=x, ylabel=y)
        axis.legend(fontsize=8)
    fig.suptitle("Synthetic measurements: training partition only", fontweight="bold")
    fig.savefig(output / "training_measurements.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(19, 8), layout="constrained")
    plot_tree(model.named_steps["tree"], feature_names=feature_names, class_names=["No failure", "Failure"],
              filled=True, impurity=False, proportion=True, precision=2, fontsize=7, ax=axis)
    axis.set_title("Depth-3 tree: node values are class-weighted training proportions")
    fig.savefig(output / "decision_tree.png", dpi=160)
    plt.close(fig)


def main():
    start = time.perf_counter()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    data, audit = load_data()
    train, test = split_source_order(data)
    model, predictions, comparison = evaluate(train, test)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    split = [{"split": name, "rows": len(frame), "udi_start": int(frame["UDI"].iloc[0]),
              "udi_end": int(frame["UDI"].iloc[-1]), "failures": int(frame[TARGET].sum()),
              "failure_pct": float(100 * frame[TARGET].mean())} for name, frame in [("train", train), ("holdout", test)]]
    pd.DataFrame(split).to_csv(output / "split_summary.csv", index=False)
    feature_names = [name.replace("numeric__", "").replace("type__", "")
                     for name in model.named_steps["preprocessing"].get_feature_names_out()]
    pd.DataFrame({"feature": feature_names, "importance": model.named_steps["tree"].feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    (output / "decision_rules.txt").write_text(export_text(model.named_steps["tree"], feature_names=feature_names,
                                                         decimals=2), encoding="utf-8")
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    save_charts(train, predictions, comparison, model, feature_names, output)
    metrics = {"source_is_synthetic": True, "rows": len(data), "features": FEATURES,
               "excluded_ids": ["UDI", "Product ID"], "excluded_auxiliary_targets": FAILURE_FLAGS,
               "split": split, "random_state": SEED, "max_depth": 3, "min_samples_leaf": 20,
               "class_weight": "balanced", "actual_tree_depth": int(model.named_steps["tree"].get_depth()),
               "tree_leaves": int(model.named_steps["tree"].get_n_leaves()),
               "scores": comparison.to_dict(orient="records"),
               "note": "Source-order holdout of synthetic current-condition labels; no dated future warning horizon or calibrated risk.",
               "run_seconds": round(time.perf_counter() - start, 3)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print("\nSplit:", split)
    print("Main label / auxiliary flag-union disagreements retained:", audit["flag_union_disagreements"])


if __name__ == "__main__":
    main()
