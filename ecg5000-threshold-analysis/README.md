ECG5000 — False-Alarm-Budget Threshold Analysis (+ Stretch Baselines)
Post-hoc analysis of an already-trained Dense Autoencoder for ECG5000 anomaly detection. This repo does not train or retrain the Autoencoder — it only takes reconstruction errors from the existing model and evaluates a new threshold-selection rule against the previous fixed-percentile baseline.

Original contribution (core of this project): the false-alarm-budget threshold rule T*(alpha) — see threshold_analysis.py. Instead of a fixed percentile, the threshold is chosen as the smallest value on validation-normal that keeps the false-positive rate under an explicit, justified budget alpha. This is evaluated across five budgets and compared to the old baseline_p95 threshold, including a bootstrap stability check (the validation-normal set is only n=59, so threshold stability itself is part of the analysis).

Stretch (bonus) — added on top, not replacing the contribution above: stretch_baselines.py fits a One-Class SVM and an Isolation Forest (both trained on normal-only data, like the Autoencoder) and evaluates them with the same false-alarm-budget rule, so the comparison across all three families (Autoencoder / OCSVM / Isolation Forest) uses one consistent threshold-selection method — the point of comparison is your own rule applied to different anomaly scores, not three unrelated pipelines.

What this does
Computes reconstruction MSE for validation-normal and test data using an already-trained model (passed in, not created here).
Implements T*(alpha): the smallest threshold such that the false-positive rate on validation-normal is <= alpha, for alpha ∈ {0.10, 0.07, 0.05, 0.035, 0.02}.
Compares 6 methods (baseline_p95 + 5 alpha budgets) on the held-out test set: precision, recall, F1, accuracy, FP/FN counts, confusion matrix, and per-subtype recall (ECG5000 original classes 2–5).
Produces 5 required plots (threshold vs P/R/F1, threshold vs FP/FN, threshold vs per-subtype recall, error histogram with both thresholds marked, PR curve with each threshold labeled).
Runs a 1000-sample bootstrap over validation-normal errors to check how stable the alpha=0.035 threshold is, given the small validation set.
Prints an “Interpretation prompts” section with raw numbers only — no conclusions are pre-written; you fill in the reasoning in your report.
No test labels or test metrics are used to select any threshold. Every threshold comes from validation-normal only.

Files
text
ecg5000-threshold-analysis/
├── README.md
├── requirements.txt
├── threshold_analysis.py   # core original contribution: false-alarm-budget rule
├── stretch_baselines.py    # bonus: OCSVM + Isolation Forest, same rule applied
└── .gitignore
Requirements / assumed variables
This script assumes you already have a trained model and preprocessed splits from your baseline notebook. Before running it, the following must exist in your Python session:


Variable	Shape / type	Meaning
model	trained Keras model	Dense Autoencoder, already trained
X_val_normal_scaled	(n_val, 140)	scaled validation-normal set
X_test_scaled	(4500, 140)	scaled full test set
y_test_binary	(4500,)	0 = Normal, 1 = Abnormal
y_test_original	(4500,)	original ECG5000 classes 1…5
threshold_base	float	previous threshold (95th pct val-normal)
If any of these are missing, the script raises a NameError with a clear message instead of guessing or retraining anything.

For stretch_baselines.py you additionally need:


Variable	Shape / type	Meaning
X_train_normal_scaled	(n_train, 140)	scaled train-normal set (same one the AE was trained on)
stretch_baselines.py imports false_alarm_budget_threshold and evaluate_threshold_on_test directly from threshold_analysis.py — it does not redefine the rule, it reuses it.

How to run
Option A — inside the same notebook (recommended)
Paste the contents of threshold_analysis.py into a new cell after the cells that build model, X_val_normal_scaled, etc., and run it. It will execute automatically via the if __name__ == "__main__": block.

Option B — import as a module
python
import sys
sys.path.append("/path/to/ecg5000-threshold-analysis")
from threshold_analysis import run_analysis

results = run_analysis(
    model=model,
    X_val_normal_scaled=X_val_normal_scaled,
    X_test_scaled=X_test_scaled,
    y_test_binary=y_test_binary,
    y_test_original=y_test_original,
    threshold_base=threshold_base,
)
results is a dict containing alpha_table, final_table, per_method_eval, bootstrap_thresholds, and the raw error arrays, in case you want to inspect them further or export to CSV for the report.

Running the stretch comparison (OCSVM / Isolation Forest)
After running the Autoencoder analysis above (so final_table exists), run in the same session:

python
from stretch_baselines import run_stretch_comparison

stretch_results = run_stretch_comparison(
    X_train_normal_scaled=X_train_normal_scaled,
    X_val_normal_scaled=X_val_normal_scaled,
    X_test_scaled=X_test_scaled,
    y_test_binary=y_test_binary,
    y_test_original=y_test_original,
    autoencoder_final_table=results["final_table"],  # optional, for the combined table/plot
)
This fits both baselines on X_train_normal_scaled only (no test leakage, same discipline as the Autoencoder), scores validation-normal and test with each model, applies the same T*(alpha) rule per baseline, and prints a combined table (Autoencoder + OCSVM + Isolation Forest) sorted by test F1, plus a bar-chart comparison at alpha=0.035.

Reproducibility
Random seed fixed to 42 (np.random.seed + a dedicated np.random.RandomState(42) used for the bootstrap resampling).
No stochastic model behavior is introduced — model.predict is deterministic given fixed weights.
Bootstrap resampling of validation-normal errors is the only randomized step in this script.
Setting this up as a git repo
bash
cd ecg5000-threshold-analysis
git init
git add .
git commit -m "Add false-alarm-budget threshold analysis (no retraining)"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
Scope / non-goals
This repo intentionally does not include: Autoencoder retraining, new Autoencoder architectures (Conv1D, LSTM, attention), or tuning OCSVM/Isolation Forest hyperparameters against test performance. The Autoencoder stays exactly as trained; the false-alarm-budget rule is the original contribution; OCSVM/Isolation Forest are the assignment’s explicitly allowed stretch/bonus comparison, evaluated with that same rule rather than a separate, untethered threshold choice.

