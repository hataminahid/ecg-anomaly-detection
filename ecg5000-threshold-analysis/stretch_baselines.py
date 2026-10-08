# -*- coding: utf-8 -*-

#Stretch (bonus) — One-Class SVM & Isolation Forest baselines


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from threshold_analysis import (
    SEED,
    ALPHA_VALUES,
    false_alarm_budget_threshold,
    evaluate_threshold_on_test,
    reconstruction_mse,
)

np.random.seed(SEED)

REQUIRED_STRETCH_VARS = [
    "X_train_normal_scaled", "X_val_normal_scaled", "X_test_scaled",
    "y_test_binary", "y_test_original",
]


def _check_required_vars(ns):
    missing = [v for v in REQUIRED_STRETCH_VARS if v not in ns]
    if missing:
        raise NameError(
            "متغیرهای زیر پیدا نشدند: " + ", ".join(missing) +
            ". این فایل مدلی را آموزش نمی‌دهد جز خود One-Class SVM / Isolation "
            "Forest (که این‌ها دقیقاً همان Stretch بونوس مجاز هستند، نه Autoencoder جدید)."
        )


#  Fit One-Class SVM and Isolation Forest — normal-only, like the AE
def fit_ocsvm(X_train_normal_scaled, nu=0.05, gamma="scale", seed=SEED):
    """nu ~ upper bound on the fraction of training errors / support vectors,
    kept small and fixed (not tuned on test)."""
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train_normal_scaled)
    return model


def fit_isolation_forest(X_train_normal_scaled, n_estimators=200, seed=SEED):
    model = IsolationForest(
        n_estimators=n_estimators,
        random_state=seed,
        contamination="auto",
    )
    model.fit(X_train_normal_scaled)
    return model


def ocsvm_anomaly_scores(model, X):
    """Higher = more anomalous. sklearn's decision_function is higher for
    inliers, so we flip the sign to match the AE convention (higher error
    = more anomalous)."""
    return -model.decision_function(X)


def isoforest_anomaly_scores(model, X):
    """Same convention: higher = more anomalous."""
    return -model.decision_function(X)



# Apply the SAME false-alarm-budget rule to each baseline's scores
def evaluate_baseline_family(name_prefix, val_scores, test_scores,
                              y_test_binary, y_test_original,
                              alpha_values=ALPHA_VALUES):
    rows = []
    per_method_eval = {}
    for alpha in alpha_values:
        T, val_fpr = false_alarm_budget_threshold(val_scores, alpha)
        res = evaluate_threshold_on_test(test_scores, y_test_binary, y_test_original, T)
        method_name = f"{name_prefix}_alpha_{alpha}"
        per_method_eval[method_name] = res
        rows.append({
            "method": method_name,
            "alpha": alpha,
            "threshold": T,
            "validation_FPR": val_fpr,
            "test_precision": res["precision"],
            "test_recall": res["recall"],
            "test_F1": res["f1"],
            "test_accuracy": res["accuracy"],
            "test_FP": res["FP"],
            "test_FN": res["FN"],
            "recall_class_2": res["recall_class_2"],
            "recall_class_3": res["recall_class_3"],
            "recall_class_4": res["recall_class_4"],
            "recall_class_5": res["recall_class_5"],
        })
    return pd.DataFrame(rows), per_method_eval


# Main stretch routine
def run_stretch_comparison(X_train_normal_scaled, X_val_normal_scaled, X_test_scaled,
                            y_test_binary, y_test_original,
                            autoencoder_final_table=None,
                            alpha_values=ALPHA_VALUES, seed=SEED):

    y_test_binary = np.asarray(y_test_binary)
    y_test_original = np.asarray(y_test_original)

    # fit baselines on train-normal only  no test leakage
    ocsvm = fit_ocsvm(X_train_normal_scaled, seed=seed)
    isoforest = fit_isolation_forest(X_train_normal_scaled, seed=seed)

    ocsvm_val_scores = ocsvm_anomaly_scores(ocsvm, X_val_normal_scaled)
    ocsvm_test_scores = ocsvm_anomaly_scores(ocsvm, X_test_scaled)

    if_val_scores = isoforest_anomaly_scores(isoforest, X_val_normal_scaled)
    if_test_scores = isoforest_anomaly_scores(isoforest, X_test_scaled)

    print("OCSVM val-normal scores    : mean=%.6f std=%.6f" %
          (ocsvm_val_scores.mean(), ocsvm_val_scores.std()))
    print("OCSVM test scores          : mean=%.6f std=%.6f" %
          (ocsvm_test_scores.mean(), ocsvm_test_scores.std()))
    print("IsoForest val-normal scores: mean=%.6f std=%.6f" %
          (if_val_scores.mean(), if_val_scores.std()))
    print("IsoForest test scores      : mean=%.6f std=%.6f" %
          (if_test_scores.mean(), if_test_scores.std()))

    #  apply the SAME false-alarm-budget rule your original contribution
    ocsvm_table, ocsvm_eval = evaluate_baseline_family(
        "ocsvm", ocsvm_val_scores, ocsvm_test_scores,
        y_test_binary, y_test_original, alpha_values
    )
    isoforest_table, isoforest_eval = evaluate_baseline_family(
        "isoforest", if_val_scores, if_test_scores,
        y_test_binary, y_test_original, alpha_values
    )

    print("\n=== One-Class SVM — false-alarm-budget results ===")
    print(ocsvm_table.to_string(index=False))
    print("\n=== Isolation Forest — false-alarm-budget results ===")
    print(isoforest_table.to_string(index=False))

    # combined comparison table AE + OCSVM + IsoForest
    combined_frames = [ocsvm_table, isoforest_table]
    if autoencoder_final_table is not None:
        ae_table = autoencoder_final_table.copy()
        ae_table["method"] = "autoencoder_" + ae_table["method"].astype(str)
        combined_frames.insert(0, ae_table)

    combined_table = pd.concat(combined_frames, ignore_index=True, sort=False)
    combined_table = combined_table.sort_values("test_F1", ascending=False).reset_index(drop=True)

    print("\n=== Combined comparison — all families, sorted by test F1 ===")
    cols_to_show = ["method", "alpha", "threshold", "validation_FPR",
                     "test_precision", "test_recall", "test_F1", "test_FP", "test_FN",
                     "recall_class_2", "recall_class_3", "recall_class_4", "recall_class_5"]
    cols_to_show = [c for c in cols_to_show if c in combined_table.columns]
    print(combined_table[cols_to_show].to_string(index=False))

    # ---- plot: F1 comparison across all methods at alpha=0.035 ----
    _plot_family_comparison_at_alpha(combined_table, alpha=0.035)

    return {
        "ocsvm_model": ocsvm,
        "isoforest_model": isoforest,
        "ocsvm_table": ocsvm_table,
        "isoforest_table": isoforest_table,
        "combined_table": combined_table,
    }


def _plot_family_comparison_at_alpha(combined_table, alpha):
    subset = combined_table[np.isclose(combined_table["alpha"], alpha)]
    if subset.empty:
        print(f"No rows found at alpha={alpha} for family comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(subset))
    width = 0.25
    ax.bar(x - width, subset["test_precision"], width, label="Precision")
    ax.bar(x, subset["test_recall"], width, label="Recall")
    ax.bar(x + width, subset["test_F1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["method"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Family Comparison at alpha={alpha} (validation-normal false-alarm budget)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# Direct-run mode
if __name__ == "__main__":
    _check_required_vars(globals())
    _ae_final_table = globals().get("final_table", None)
    stretch_results = run_stretch_comparison(
        X_train_normal_scaled=X_train_normal_scaled,
        X_val_normal_scaled=X_val_normal_scaled,
        X_test_scaled=X_test_scaled,
        y_test_binary=y_test_binary,
        y_test_original=y_test_original,
        autoencoder_final_table=_ae_final_table,
    )
