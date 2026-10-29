# Project 10: folder structure and copyable files

Files are already created. Each hand-authored file has a separate code block below. The small original UCI synthetic CSV is bundled; main.py can download and checksum-verify it if absent. Running main.py recreates generated artifacts. This current-condition experiment has no future warning horizon and no dependency on another project.

```text
projects/10-predictive-maintenance-explorer/
  main.py
  test_maintenance.py
  requirements.txt
  .gitignore
  .gitattributes
  README.md
  PROJECT_FILES.md
  data/
    ai4i2020.csv  (unchanged public SYNTHETIC source)
    SOURCE.md
  outputs/
    metrics.json
    data_audit.json
    split_summary.csv
    model_comparison.csv
    holdout_predictions.csv
    feature_importance.csv
    decision_rules.txt
    confusion_matrices.png
    training_measurements.png
    decision_tree.png
    verification.md
```

## main.py

```python
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
```

## test_maintenance.py

```python
"""Checks leakage boundaries, source-order isolation and classification arithmetic."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from main import FAILURE_FLAGS, FEATURES, NUMERIC, TARGET, binary_metrics, evaluate, load_data, split_source_order


class MaintenanceChecks(unittest.TestCase):
    def test_source_labels_and_integrity(self):
        data, audit = load_data()
        self.assertEqual(len(data), 10000)
        self.assertEqual(int(data[TARGET].sum()), 339)
        self.assertEqual(audit["flag_union_disagreements"], 27)
        self.assertTrue(audit["source_is_synthetic"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.csv"
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(path)

    def test_source_order_split(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        self.assertEqual((len(train), len(test)), (8000, 2000))
        self.assertEqual((train["UDI"].iloc[-1], test["UDI"].iloc[0]), (8000, 8001))
        self.assertFalse(set(train["UDI"]) & set(test["UDI"]))
        self.assertEqual((int(train[TARGET].sum()), int(test[TARGET].sum())), (300, 39))

    def test_hand_calculated_positive_class_metrics(self):
        result = binary_metrics([0, 0, 0, 1, 1], [0, 1, 0, 1, 0])
        self.assertEqual((result["true_negative"], result["false_positive"], result["false_negative"], result["true_positive"]), (2, 1, 1, 1))
        self.assertEqual(result["accuracy"], 0.6)
        self.assertEqual(result["failure_precision"], 0.5)
        self.assertEqual(result["failure_recall"], 0.5)
        self.assertEqual(result["failure_f1"], 0.5)
        baseline = binary_metrics([0, 0, 0, 1, 1], [0, 0, 0, 0, 0])
        self.assertEqual(baseline["failure_recall"], 0)
        self.assertEqual(baseline["failure_precision"], 0)

    def test_ids_and_failure_modes_cannot_change_predictions(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        first, before, _ = evaluate(train, test)
        for frame in [train, test]:
            frame[FAILURE_FLAGS] = 1 - frame[FAILURE_FLAGS]
            frame["UDI"] += 50000
            frame["Product ID"] = "IGNORED"
        second, after, _ = evaluate(train, test)
        np.testing.assert_array_equal(before["shallow_tree"], after["shallow_tree"])
        self.assertFalse(set(FEATURES) & set(FAILURE_FLAGS + ["UDI", "Product ID", TARGET]))
        self.assertEqual(first.named_steps["tree"].get_depth(), second.named_steps["tree"].get_depth())

    def test_holdout_labels_do_not_fit_preprocessing_or_model(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        first, before, _ = evaluate(train, test)
        changed = test.copy()
        changed[TARGET] = 1 - changed[TARGET]
        second, after, _ = evaluate(train, changed)
        np.testing.assert_array_equal(before["shallow_tree"], after["shallow_tree"])
        np.testing.assert_allclose(before["weighted_tree_score"], after["weighted_tree_score"])
        medians = first.named_steps["preprocessing"].named_transformers_["numeric"].statistics_
        np.testing.assert_allclose(medians, train[NUMERIC].median())
        self.assertLessEqual(second.named_steps["tree"].get_depth(), 3)

    def test_confusion_counts_and_reproducibility(self):
        data, _ = load_data()
        train, test = split_source_order(data)
        _, predictions, scores = evaluate(train, test)
        _, again, _ = evaluate(train, test)
        np.testing.assert_array_equal(predictions["shallow_tree"], again["shallow_tree"])
        for row in scores.itertuples(index=False):
            self.assertEqual(row.true_negative + row.false_positive + row.false_negative + row.true_positive, len(test))
            self.assertEqual(row.false_negative + row.true_positive, 39)
        self.assertTrue(predictions["weighted_tree_score"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
```

## requirements.txt

```text
numpy==2.5.4
pandas==3.1.0
scikit-learn==1.9.1
matplotlib==3.11.2
```

## .gitignore

```text
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
.env.*
!.env.example
*.log
```

## .gitattributes

```text
data/ai4i2020.csv -text
```

## data/SOURCE.md

````markdown
# Source, license and target handling

- **Dataset:** AI4I 2020 Predictive Maintenance Dataset, UCI dataset 601.
- **Public source:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Direct download:** https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip
- **Attribution:** AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- **License:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- **Original file:** ai4i2020.csv, unchanged; 10,000 rows and 14 columns.
- **SHA-256:** `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

This is **published synthetic data**, not real factory telemetry and not a dataset generated by this project. UCI describes rule-based failure modes and random-walk temperature measurements. The original generator seed is not claimed; source checksum and model seed 42 provide reproducibility of this exercise.

The primary Machine failure label has 339 positives. The first 8,000 source rows contain 300 positives, and the final 2,000 contain 39. A contiguous source-order split limits mixing across the ordered synthetic sequence, but cannot establish real calendar chronology, a future warning horizon or independent machine histories.

Audit found 27 disagreements between Machine failure and the OR of the five auxiliary flags: nine main-label positives without a flag, and 18 flagged rows with a zero main label. All published main labels are retained, with no redefinition or silent repair. The discrepancy needs source-owner clarification.

The model uses Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm] and Tool wear [min]. UDI, Product ID and all five auxiliary outcomes are excluded. No temperatures are converted to Celsius, no missing measurements are synthesized, and no new observations or timestamps are invented.
````

## README.md

````markdown
# Predictive Maintenance Explorer

A source-order equipment-condition classification experiment using a majority baseline and an interpretable class-weighted decision tree, with explicit positive-class errors and source-label reconciliation.

> Prediction boundary: the public AI4I data is synthetic and labels current conditions. The experiment does not measure advance warning, remaining useful life or real-equipment reliability.

## Contents

- [Overview](#overview)
- [Dataset and data contract](#dataset-and-data-contract)
- [Architecture and components](#architecture-and-components)
- [Processing pipeline](#processing-pipeline)
- [Configuration and metric definitions](#configuration-and-metric-definitions)
- [Recorded results and evidence](#recorded-results-and-evidence)
- [Interpretation and operational use](#interpretation-and-operational-use)
- [Running locally](#running-locally)
- [Validation and reproducibility](#validation-and-reproducibility)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Limitations and next steps](#limitations-and-next-steps)
- [Further technical documentation](#further-technical-documentation)

## Overview

Which measurement observations indicate **current failure conditions** that an operations team might investigate?

## Dataset and data contract

- Public source: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), **10,000 synthetic observations and 14 source columns**.
- Attribution: AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. [DOI: 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The unchanged [original CSV](data/ai4i2020.csv) is bundled. [data/SOURCE.md](data/SOURCE.md) records source, checksum, units and label handling.
- There are **9,661 non-failure labels and 339 failure labels**: overall failure prevalence **3.39%**. All observations are retained and the source has no missing values.
- The source generator's random seed is unknown to this project. The data is fixed by checksum; **42** is the decision tree's seed, not a claim about generating the original simulation.

| Field | Meaning / use |
|---|---|
| Type | Product quality variant L/M/H; categorical input |
| Air temperature [K] | Ambient measurement in kelvin; numeric input |
| Process temperature [K] | Process measurement in kelvin; numeric input |
| Rotational speed [rpm] | Rotation measurement; numeric input |
| Torque [Nm] | Torque measurement; numeric input |
| Tool wear [min] | Reported wear in minutes; numeric input, not remaining life |
| Machine failure | Original binary target; positive class is 1 |
| UDI / Product ID | Source identifiers, retained for traceability but excluded from features |
| TWF / HDF / PWF / OSF / RNF | Auxiliary tool-wear, heat-dissipation, power, overstrain and random-failure outcome flags; all excluded |

**Label audit:** 27 rows disagree between the main target and the union of the five auxiliary flags: nine main-label positives have no flag, and 18 flagged rows have a zero main label. The project preserves the published main target. It does not silently regenerate labels from the flags or train with those outcome columns.

## Architecture and components

```mermaid
flowchart LR
    N0["Original synthetic AI4I source"]
    N1["Target and auxiliary-label audit"]
    N2["UDI-ordered 8000/2000 split"]
    N3["Training-only encoding"]
    N4["Majority baseline and depth-3 tree"]
    N5["Failure confusion counts and rules"]
    N0 --> N1 --> N2 --> N3 --> N4 --> N5
```

| Component | Entry points or assets | Responsibility |
|---|---|---|
| [main.py](main.py) | load_data / split_source_order | Verify source bytes, label disagreements and the fixed UDI-order split. |
| [main.py](main.py) | build_tree | Encode Type and fit the bounded class-weighted tree without identifier/outcome features. |
| [main.py](main.py) | binary_metrics / evaluate | Compute positive-class precision/recall/F1, balanced accuracy and confusion counts. |
| [main.py](main.py) | save_charts | Export training-only measurement exploration, the fitted tree and holdout confusion matrices. |
| [test_maintenance.py](test_maintenance.py) | Automated checks | Verify hand-calculated metrics, identifier/label exclusion, train-only fitting and repeatability. |

## Processing pipeline

1. Verify source bytes, expected shape, sequence, missingness and label codes. Record target/flag discrepancies without altering labels.
2. Preserve source order: train on **UDI 1-8,000** and test on **UDI 8,001-10,000**. The source contains ordered random-walk signals, so a random row shuffle is avoided. This is an ordered simulation holdout, not a dated operational forecast.
3. Whitelist the six legitimate inputs. Fit numeric median and categorical most-frequent imputation on training data only, then one-hot encode Type. Original inputs are complete, so imputation is inactive. Trees do not need feature scaling; measurement thresholds remain in original units.
4. Fit a majority-class baseline and one [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) with fixed `max_depth=3`, `min_samples_leaf=20`, `class_weight='balanced'` and seed 42. No parameter/threshold search uses the holdout.
5. Score failure precision, recall and F1, plus accuracy, balanced accuracy and confusion counts. Treat class 1 consistently as failure; zero predicted positives give precision/recall/F1 of zero for the baseline.
6. Export test predictions, audit/split summaries, feature importances, text rules and figures. Measurement exploration charts use the training partition only.

Class balancing gives rare failures more influence while fitting. The exported `weighted_tree_score` is a weighted leaf-class fraction; it is **not a calibrated real-world failure probability**. The default class decision is used without tuning on test labels.

## Configuration and metric definitions

| Setting or measure | Definition |
|---|---|
| Inputs | Type plus five measurements: air/process temperature, rotation speed, torque and tool wear |
| Excluded fields | UDI, Product ID and all five auxiliary failure-mode labels |
| Positive class | Published `Machine failure == 1` |
| Tree | `max_depth=3`, `min_samples_leaf=20`, `class_weight="balanced"`, seed 42 |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | Harmonic mean of positive-class precision and recall |
| Balanced accuracy | Mean of failure recall and non-failure recall |

The majority baseline predicts no failures and receives zero positive-class precision/recall/F1 under the implemented zero-division convention. Tree scores are weighted leaf-class fractions rather than calibrated failure probabilities. No test-set threshold search or relabeling from auxiliary outcome flags is performed.

## Recorded results and evidence

The fixed source-order split contains **300 failures / 8,000 training rows (3.75%)** and **39 failures / 2,000 test rows (1.95%)**. This prevalence shift is retained rather than hidden by stratifying/shuffling the sequence.

| Method | Accuracy | Balanced accuracy | Failure precision | Failure recall | Failure F1 |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 98.05% | 50.00% | 0.00% | 0.00% | 0.0000 |
| Shallow tree | 91.40% | 78.02% | **13.66%** | **64.10%** | **0.2252** |

These values were computed by running the project. Accuracy alone would favor a baseline that misses every failure. The tree retrieves some failures but produces many false alerts; this is a limited synthetic triage experiment rather than a validated maintenance system.

| Tree holdout outcome | Count |
|---|---:|
| Correct non-failure labels (TN) | 1,803 |
| Non-failures flagged as failures (FP) | 158 |
| Failures missed (FN) | 14 |
| Failures detected (TP) | 25 |

The tree flags **183 observations**, of which **25** match the main failure label. It detects **25 of 39 failures** and misses 14. No actual machine inspection, downtime reduction or cost saving was measured.

![Actual majority-baseline and tree confusion matrices](outputs/confusion_matrices.png)

![Synthetic training measurements and their current labels](outputs/training_measurements.png)

The training chart shows associations within the synthetic generator's measurements, not real-factory causal effects. UCI's simulation rules contribute to the observed patterns.

![Fitted shallow tree](outputs/decision_tree.png)

The fitted tree has **depth 3 and eight leaves**. In its plot, `value` shows **class-weighted training proportions**; it must not be read as raw failure prevalence. Node sample percentages describe the training sample, while the 50/50 root values arise from class weighting. For rule inspection, [plain-text rules](outputs/decision_rules.txt) are easier to inspect than a zoomed-out figure. `class: 1` means a failure flag under this model.

Evidence: [metrics](outputs/metrics.json), [comparison table](outputs/model_comparison.csv), [test predictions](outputs/holdout_predictions.csv), [split summary](outputs/split_summary.csv), [source audit](outputs/data_audit.json), [feature importance](outputs/feature_importance.csv) and [verification record](outputs/verification.md).

Feature importance is the tree's training impurity-reduction measure. It is neither a causal effect nor proof that an unused feature is irrelevant to actual equipment.

## Interpretation and operational use

- **Evaluate failures directly:** the 98.05%-accurate baseline detects zero failures. Keep recall, precision, F1 and counts alongside accuracy in any maintenance experiment.
- **Account for false-alert workload:** the shallow tree detects 25 failures but flags another 158 non-failures and misses 14 failures. Use those counts to discuss investigation capacity; real operating thresholds would require independent validation and actual error costs.
- **Resolve data meaning before operational use:** retain the 27 main-label/flag disagreements visibly and request source clarification. A genuine advance-warning project would also need dated measurement histories, independent machine identities and a defined future horizon. No future warning performance is established here.

## Running locally

Prerequisite: **Python 3.12+** and a terminal; verified with Python 3.12. No paid API, cloud account, GPU, model runtime, database server or other project folder is needed. From the repository root, run these three commands in your Python environment:

```powershell
cd projects/10-predictive-maintenance-explorer
python -m pip install -r requirements.txt
python main.py
```

The source is bundled for offline data use. If absent, the official small archive is downloaded and checksum-verified. A changed source fails explicitly; `.gitattributes` preserves original CSV bytes across Git checkouts. No large model binary is necessary.

Optional verification, separate from launching: `python -m unittest -v`. This is a local script and saved figures, so no web service is required.

## Validation and reproducibility

The automated suite contains **6 tests**. It checks the following implementation boundaries:

Test implementation: [test_maintenance.py](test_maintenance.py).

- Source labels and integrity.
- Source order split.
- Hand calculated positive class metrics.
- Ids and failure modes cannot change predictions.
- Holdout labels do not fit preprocessing or model.
- Confusion counts and reproducibility.

From the project directory, run:

```powershell
python -m unittest -v
```

Recorded execution and any browser-review evidence is retained in [outputs/verification.md](outputs/verification.md). Reported figures correspond to saved execution artifacts; running a report generator can overwrite its generated outputs. Package requirements, source checksums and fixed parameters define the reproducibility boundary; elapsed times and some model/solver outputs can vary across environments.

## Project structure

```text
10-predictive-maintenance-explorer/
  main.py
  PROJECT_FILES.md
  README.md
  requirements.txt
  test_maintenance.py
  data/
    Bundled inputs, provenance and prepared data
  outputs/
    Recorded metrics, audit files, tables and figures
```

Generated environments, database files, model artifacts and compiled assets follow the project's [.gitignore](.gitignore). The [copyable source bundle](PROJECT_FILES.md) mirrors the README and contains individual file blocks.

## Technology stack

Python, NumPy, pandas, scikit-learn Pipeline/ColumnTransformer, SimpleImputer, OneHotEncoder, DummyClassifier, DecisionTreeClassifier and classification metrics; matplotlib; standard-library unittest/hashlib/urllib/zipfile. Direct dependencies are pinned in [requirements.txt](requirements.txt).

## Limitations and next steps

- The source is synthetic, generated using designed failure relationships; the model partly learns that simulation. Results do not validate performance on actual machinery.
- The target is a current-condition label. There is no future warning interval, remaining useful life or calendar-dated forecasting task.
- Only 39 holdout failures support the reported failure metrics. One fixed split is limited evidence; no confidence intervals, calibration or production validation were performed.
- Ordered holdout reduces random mixing of sequence-related measurements but does not prove independent machine or temporal generalization. Actual dates and independent equipment histories are unavailable.
- Source main-label/auxiliary-flag disagreements are retained. Errors or undocumented labeling differences can affect fitting and evaluation.
- The tree's low precision creates substantial false-alert workload. Balanced class weights and weighted scores are not real-world risk calibration or a policy chosen from actual costs.
- Fixed depth/leaf settings keep the model explainable but can miss interactions and uncommon failure modes. No test-set tuning, oversampling or alternate classifier search was performed.
- Complete source measurements mean imputation is inactive; it is a reusable training-only preprocessing step, not evidence of repaired missing telemetry.
- No automated maintenance action, real-world downtime reduction, safety guarantee, inspection outcome or financial saving was measured. Future work should begin with a suitable dated real dataset and independently defined operating criteria.

## Further technical documentation

- [Dataset attribution, licensing and source transformations](data/SOURCE.md)
- [Pinned Python dependencies](requirements.txt)
- [Recorded execution and verification evidence](outputs/verification.md)
- [Folder structure and individually copyable source files](PROJECT_FILES.md)

[Back to contents](#contents)
````
