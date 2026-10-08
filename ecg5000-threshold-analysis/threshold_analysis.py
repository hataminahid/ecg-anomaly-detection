# -*- coding: utf-8 -*-

#ECG5000 — False-Alarm-Budget Threshold Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve,
)

SEED = 42
np.random.seed(SEED)

# Sanity check — required variables must already exist in the caller's
#    namespace. We do NOT retrain or fabricate any of them.

REQUIRED_VARS = [
    "model", "X_val_normal_scaled", "X_test_scaled",
    "y_test_binary", "y_test_original", "threshold_base",
]


def _check_required_vars(ns):
    missing = [v for v in REQUIRED_VARS if v not in ns]
    if missing:
        raise NameError(
            "متغیرهای زیر در محیط اجرا پیدا نشدند: "
            + ", ".join(missing)
            + ". این اسکریپت مدل را آموزش نمی‌دهد؛ باید بعد از سلول‌های آموزش "
              "مدل موجود (که این متغیرها را می‌سازد) اجرا/import شود."
        )


# Reconstruction MSE
def reconstruction_mse(model, X):
    """error_i = mean((x_i - xhat_i)^2), X shape (n, 140)."""
    X = np.asarray(X)
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


#  False-alarm-budget threshold rule (validation-normal ONLY)
def false_alarm_budget_threshold(val_normal_errors, alpha):
    """
    T*(alpha) = smallest T such that FPR_val(T) <= alpha, where
    FPR_val(T) = mean(val_normal_errors > T).

    Candidates are the unique observed validation-normal error values
    (ascending). We return the first (smallest) candidate meeting the
    budget, to keep sensitivity as high as possible while respecting alpha.
    Test data/labels are never used here.
    """
    candidates = np.unique(val_normal_errors)
    for T in candidates:
        fpr = np.mean(val_normal_errors > T)
        if fpr <= alpha:
            return float(T), float(fpr)
    #  no candidate satisfies alpha  use the max observed value
    T = candidates[-1]
    fpr = np.mean(val_normal_errors > T)
    return float(T), float(fpr)


# Test-set metrics for a given threshold
def evaluate_threshold_on_test(test_errors, y_true_binary, y_true_original, threshold):
    y_pred = (test_errors > threshold).astype(int)

    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true_binary, y_pred)
    cm = confusion_matrix(y_true_binary, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall_per_class = {}
    for c in [2, 3, 4, 5]:
        mask = (y_true_original == c)
        n_total = int(mask.sum())
        if n_total == 0:
            recall_per_class[c] = np.nan
            continue
        n_tp = int(np.sum(mask & (y_pred == 1)))
        recall_per_class[c] = n_tp / n_total

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN": int(tn),
        "recall_class_2": recall_per_class[2],
        "recall_class_3": recall_per_class[3],
        "recall_class_4": recall_per_class[4],
        "recall_class_5": recall_per_class[5],
        "y_pred": y_pred,
    }


# Main analysis routine
ALPHA_VALUES = [0.10, 0.07, 0.05, 0.035, 0.02]


def run_analysis(model, X_val_normal_scaled, X_test_scaled,
                  y_test_binary, y_test_original, threshold_base,
                  alpha_values=ALPHA_VALUES, n_bootstrap=1000, seed=SEED):

    rng = np.random.RandomState(seed)

    # errors 
    val_normal_errors = reconstruction_mse(model, X_val_normal_scaled)
    test_errors = reconstruction_mse(model, X_test_scaled)

    y_test_binary = np.asarray(y_test_binary)
    y_test_original = np.asarray(y_test_original)

    test_normal_errors = test_errors[y_test_binary == 0]
    test_abnormal_errors = test_errors[y_test_binary == 1]

    print("val-normal errors  : n=%d mean=%.6f std=%.6f" %
          (len(val_normal_errors), val_normal_errors.mean(), val_normal_errors.std()))
    print("test errors         : n=%d mean=%.6f std=%.6f" %
          (len(test_errors), test_errors.mean(), test_errors.std()))

    #  per-alpha threshold table 
    alpha_rows = []
    alpha_thresholds = {}
    for alpha in alpha_values:
        T, val_fpr = false_alarm_budget_threshold(val_normal_errors, alpha)
        n_above = int(np.sum(val_normal_errors > T))
        alpha_thresholds[alpha] = T
        alpha_rows.append({
            "alpha": alpha,
            "selected_threshold": T,
            "validation_normal_FPR": val_fpr,
            "n_val_normal_above_threshold": n_above,
        })
    alpha_table = pd.DataFrame(alpha_rows)
    print("\n=== Per-alpha threshold table ===")
    print(alpha_table.to_string(index=False))

    #  evaluate baseline + all alpha thresholds on test 
    methods = [("baseline_p95", None, threshold_base)]
    for alpha in alpha_values:
        methods.append((f"false_alarm_budget_alpha_{alpha}", alpha, alpha_thresholds[alpha]))

    final_rows = []
    per_method_eval = {}
    for name, alpha, T in methods:
        res = evaluate_threshold_on_test(test_errors, y_test_binary, y_test_original, T)
        per_method_eval[name] = res
        val_fpr = (np.mean(val_normal_errors > T))
        final_rows.append({
            "method": name,
            "alpha": alpha if alpha is not None else np.nan,
            "threshold": T,
            "validation_FPR": val_fpr,
            "test_precision": res["precision"],
            "test_recall": res["recall"],
            "test_F1": res["f1"],
            "test_FP": res["FP"],
            "test_FN": res["FN"],
            "recall_class_2": res["recall_class_2"],
            "recall_class_3": res["recall_class_3"],
            "recall_class_4": res["recall_class_4"],
            "recall_class_5": res["recall_class_5"],
        })

    final_table = pd.DataFrame(final_rows).sort_values("threshold").reset_index(drop=True)
    print("\n=== Final comparison table (sorted by threshold) ===")
    print(final_table.to_string(index=False))

    for name, _, T in methods:
        print(f"\nConfusion matrix — {name} (threshold={T:.6f}):")
        print(per_method_eval[name]["confusion_matrix"])

    #  Plots
    _plot_threshold_vs_prf(final_table)
    _plot_threshold_vs_fp_fn(final_table)
    _plot_threshold_vs_subtype_recall(final_table)
    alpha_035_threshold = alpha_thresholds.get(0.035, None)
    _plot_error_histogram(test_normal_errors, test_abnormal_errors,
                           threshold_base, alpha_035_threshold)
    _plot_pr_curve_with_thresholds(test_errors, y_test_binary, final_table)

    #  Bootstrap stability analysis alpha = 0.035
    bootstrap_thresholds = _bootstrap_threshold_stability(
        val_normal_errors, alpha=0.035, n_bootstrap=n_bootstrap, rng=rng
    )
    boot_mean = np.mean(bootstrap_thresholds)
    boot_median = np.median(bootstrap_thresholds)
    boot_std = np.std(bootstrap_thresholds)
    boot_ci_lo, boot_ci_hi = np.percentile(bootstrap_thresholds, [2.5, 97.5])

    print("\n=== Bootstrap stability of threshold (alpha=0.035, n_bootstrap=%d) ===" % n_bootstrap)
    print(f"mean   = {boot_mean:.6f}")
    print(f"median = {boot_median:.6f}")
    print(f"std    = {boot_std:.6f}")
    print(f"95% percentile interval = [{boot_ci_lo:.6f}, {boot_ci_hi:.6f}]")

    plt.figure(figsize=(7, 4))
    plt.hist(bootstrap_thresholds, bins=40, alpha=0.8)
    plt.axvline(boot_mean, color="red", linestyle="--", label=f"mean={boot_mean:.5f}")
    plt.axvline(boot_ci_lo, color="black", linestyle=":", label="2.5th pct")
    plt.axvline(boot_ci_hi, color="black", linestyle=":", label="97.5th pct")
    plt.xlabel("Bootstrap threshold (alpha=0.035)")
    plt.ylabel("Count")
    plt.title(f"Bootstrap Threshold Distribution (n_bootstrap={n_bootstrap})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  Interpretation prompts — do not fabricate conclusions 
    _print_interpretation_prompts(
        final_table, val_normal_errors, test_errors,
        boot_ci_lo, boot_ci_hi, boot_std, n_val_normal=len(val_normal_errors)
    )

    return {
        "val_normal_errors": val_normal_errors,
        "test_errors": test_errors,
        "alpha_table": alpha_table,
        "final_table": final_table,
        "per_method_eval": per_method_eval,
        "bootstrap_thresholds": bootstrap_thresholds,
    }


# Plot helpers
def _plot_threshold_vs_prf(final_table):
    plt.figure(figsize=(8, 5))
    plt.plot(final_table["threshold"], final_table["test_precision"], marker="o", label="Precision")
    plt.plot(final_table["threshold"], final_table["test_recall"], marker="o", label="Recall")
    plt.plot(final_table["threshold"], final_table["test_F1"], marker="o", label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Test Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_threshold_vs_fp_fn(final_table):
    plt.figure(figsize=(8, 5))
    plt.plot(final_table["threshold"], final_table["test_FP"], marker="o", label="False Positives")
    plt.plot(final_table["threshold"], final_table["test_FN"], marker="o", label="False Negatives")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.title("Threshold vs Test FP / FN Counts")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_threshold_vs_subtype_recall(final_table):
    plt.figure(figsize=(8, 5))
    for c in [2, 3, 4, 5]:
        plt.plot(final_table["threshold"], final_table[f"recall_class_{c}"],
                  marker="o", label=f"class {c}")
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Threshold vs Recall per Abnormal Subtype (classes 2-5)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_error_histogram(test_normal_errors, test_abnormal_errors, threshold_base, alpha_035_threshold):
    plt.figure(figsize=(8, 5))
    bins = 50
    plt.hist(test_normal_errors, bins=bins, alpha=0.5, density=True, label="test-normal")
    plt.hist(test_abnormal_errors, bins=bins, alpha=0.5, density=True, label="test-abnormal")
    plt.axvline(threshold_base, color="red", linestyle="--", label=f"baseline_p95={threshold_base:.5f}")
    if alpha_035_threshold is not None:
        plt.axvline(alpha_035_threshold, color="green", linestyle="--",
                     label=f"alpha=0.035={alpha_035_threshold:.5f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Density")
    plt.title("Test Error Distribution: Normal vs Abnormal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_pr_curve_with_thresholds(test_errors, y_test_binary, final_table):
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test_binary, test_errors)

    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, label="PR curve (all thresholds)")

    for _, row in final_table.iterrows():
        plt.scatter(row["test_recall"], row["test_precision"], s=60)
        plt.annotate(
            row["method"].replace("false_alarm_budget_alpha_", "a="),
            (row["test_recall"], row["test_precision"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with Evaluated Thresholds Labeled")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Bootstrap stability
def _bootstrap_threshold_stability(val_normal_errors, alpha, n_bootstrap, rng):
    n = len(val_normal_errors)
    thresholds = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample = rng.choice(val_normal_errors, size=n, replace=True)
        T, _ = false_alarm_budget_threshold(sample, alpha)
        thresholds[b] = T
    return thresholds

# Interpretation prompts  raw numbers only, no conclusions written here

def _print_interpretation_prompts(final_table, val_normal_errors, test_errors,
                                   boot_ci_lo, boot_ci_hi, boot_std, n_val_normal):
    print("\n### Interpretation prompts — do not fabricate conclusions ###\n")

    baseline_row = final_table[final_table["method"] == "baseline_p95"].iloc[0]

    print("1) Validation FPR vs test FP, per method (raw numbers to compare):")
    print(final_table[["method", "validation_FPR", "test_FP"]].to_string(index=False))

    print("\n2) F1 per method (raw numbers):")
    print(final_table[["method", "threshold", "test_F1"]].sort_values("test_F1", ascending=False).to_string(index=False))

    print("\n3) Recall class 5 across methods, ordered by threshold (raw numbers):")
    print(final_table[["method", "threshold", "recall_class_5"]].to_string(index=False))

    print("\n4) Recall class 3 vs class 5 vs class 2 vs class 4, per method (raw numbers):")
    print(final_table[["method", "threshold", "recall_class_2", "recall_class_3",
                        "recall_class_4", "recall_class_5"]].to_string(index=False))

    boot_width = boot_ci_hi - boot_ci_lo
    print("\n5) Bootstrap threshold interval width and validation-normal sample size:")
    print(f"   n(validation-normal) = {n_val_normal}")
    print(f"   bootstrap 95% interval = [{boot_ci_lo:.6f}, {boot_ci_hi:.6f}]")
    print(f"   interval width = {boot_width:.6f}")
    print(f"   bootstrap std  = {boot_std:.6f}")
    print(f"   baseline threshold (for reference) = {baseline_row['threshold']:.6f}")


# Direct-run mode: if this file is executed inside the same notebook/kernel
# where the required variables already exist in globals(), run immediately.

if __name__ == "__main__":
    _check_required_vars(globals())
    results = run_analysis(
        model=model,
        X_val_normal_scaled=X_val_normal_scaled,
        X_test_scaled=X_test_scaled,
        y_test_binary=y_test_binary,
        y_test_original=y_test_original,
        threshold_base=threshold_base,
    )
