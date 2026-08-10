import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_config

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from sklearn.base import clone

from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import math

from scripts.plot_helper.colors import COLORS


import seaborn as sns
import matplotlib.pyplot as plt

import json

#* ── Constants ────────────────────────────────────────────
OUTER_SPLITS = 5
OUTER_REPEATS = 50
INNER_SPLITS = 4

N_PERMUTATIONS = 1000
PERM_OUTER_SPLITS = 5
PERM_OUTER_REPEATS = 5

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]

outer_cv = RepeatedStratifiedKFold(
    n_splits=OUTER_SPLITS,
    n_repeats=OUTER_REPEATS,
    random_state=42
)
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=42)

SLEEP_PCA_COMPONENTS = 3
STEP_PCA_COMPONENTS = 3
HR_PCA_COMPONENTS = 2

CANDIDATE_CUTOFFS = {
    "cutoff_2.5": 2.5,   
    "cutoff_3.5": 3.5,  
    "cutoff_4.0": 4.0,  
    "cutoff_4.3": 4.3,  
    "cutoff_6.4": 6.4,  
}

output_dir = (
    "output/4_Prediction_Model/3_Logistic_Regression/6_Final_Sensitivity/"
)

model_labels = {
    "A_demo": "Set A",
    "B_demo_sleep": "Set B",
    "C_demo_steps": "Set C",
    "D_demo_hr": "Set D",
    "E_all": "Set E",
}

model_order = [
    "A_demo",
    "B_demo_sleep",
    "C_demo_steps",
    "D_demo_hr",
    "E_all",
]

model_colors = {
    "Set A": COLORS["blue1"],
    "Set B": COLORS["orange1"],
    "Set C": COLORS["green1"],
    "Set D": COLORS["violet1"],
    "Set E": COLORS["teal1"],
}

#* ── Functions ────────────────────────────────────────────
def make_pca_pipeline(n_components=2):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42))
    ])

def make_logistic_regression():
    return LogisticRegression(
        l1_ratio=0.0,
        solver="liblinear",
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

def make_tuned_model(preprocessor):
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", make_logistic_regression())
    ])

    tuned_pipeline = GridSearchCV(
        estimator=full_pipeline,
        param_grid={"model__C": C_GRID},
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False
    )
    return tuned_pipeline

def evaluate_model(model, model_estimator, X_data, y_data, cross_validation, n_outer_splits):
    fold_results = []
    predictions = []

    for iteration, (train_idx, test_idx) in enumerate(cross_validation.split(X_data, y_data)):

        repeat = iteration // n_outer_splits +1
        fold_within_repeat = iteration % n_outer_splits +1

        X_train = X_data.iloc[train_idx]
        X_test = X_data.iloc[test_idx]

        y_train = y_data.iloc[train_idx]
        y_test = y_data.iloc[test_idx]

        model_estimator.fit(X_train, y_train)

        y_probability = model_estimator.predict_proba(X_test)[:, 1]
        y_predicted = model_estimator.predict(X_test)

        # Guard: AUC undefined if test fold has only one class present
        if y_test.nunique() >= 2:
            fold_auc = roc_auc_score(y_test, y_probability)
        else:
            fold_auc = np.nan

        sensitivity = recall_score(
            y_test,
            y_predicted,
            pos_label=1,
            zero_division=np.nan
        )

        specificity = recall_score(
            y_test,
            y_predicted,
            pos_label=0,
            zero_division=np.nan
        )
        # Extract tuning information.
        if isinstance(model_estimator, GridSearchCV):
            selected_C = model_estimator.best_params_["model__C"]
            inner_cv_best_score = model_estimator.best_score_

            fitted_logistic_model = (
                model_estimator
                .best_estimator_
                .named_steps["model"]
            )

            fold_coefs = fitted_logistic_model.coef_[0]
            max_abs_coef = np.max(np.abs(fold_coefs))
            possible_separation = max_abs_coef > 5

        else:
            # Dummy baseline
            selected_C = np.nan
            inner_cv_best_score = np.nan
            max_abs_coef = np.nan
            possible_separation = np.nan

        fold_results.append({
            "model": model,
            "repeat": repeat,
            "fold": fold_within_repeat,
            "fold_uid": iteration,
            "auc": fold_auc,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_predicted),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "n_pos_test": int(y_test.sum()),
            "n_neg_test": int((y_test == 0).sum()),
            "selected_C": selected_C,
            "inner_cv_best_score": inner_cv_best_score,
            "max_abs_coef": max_abs_coef,
            "possible_separation": possible_separation
        })

        predictions.append(pd.DataFrame({
            "model": model,
            "repeat": repeat,
            "fold": fold_within_repeat,
            "fold_uid": iteration,
            "participant_index": X_test.index.to_numpy(),
            "observed": y_test.to_numpy(),
            "predicted_prob": y_probability,
            "predicted_class": y_predicted
        }))


    return pd.DataFrame(fold_results), pd.concat(predictions, ignore_index=True)

 
def _fit_one_permutation(
    estimator,
    X_data,
    y_shuffled,
    cv_seed,
    n_outer_splits,
    n_outer_repeats,
):
    """
    Run one permutation's nested CV and return its participant-level AUC
    (or np.nan if the shuffled test folds only ever contain one class).
    """
    perm_cv = RepeatedStratifiedKFold(
        n_splits=n_outer_splits,
        n_repeats=n_outer_repeats,
        random_state=cv_seed,
    )
 
    permutation_predictions = []
 
    for train_idx, test_idx in perm_cv.split(X_data, y_shuffled):
        X_train = X_data.iloc[train_idx]
        X_test = X_data.iloc[test_idx]
        y_train = y_shuffled.iloc[train_idx]
        y_test = y_shuffled.iloc[test_idx]
 
        fitted_estimator = clone(estimator)
        # Force single-threaded inner GridSearchCV fitting: the outer
        # permutation loop is already parallelized across processes, so
        # letting GridSearchCV also spawn n_jobs=-1 workers per fold would
        # oversubscribe cores (nested parallelism) without changing results,
        # only slowing things down.
        if isinstance(fitted_estimator, GridSearchCV):
            fitted_estimator.set_params(n_jobs=1)
        fitted_estimator.fit(X_train, y_train)
 
        y_prob = fitted_estimator.predict_proba(X_test)[:, 1]
 
        permutation_predictions.append(
            pd.DataFrame({
                "participant_index": X_test.index.to_numpy(),
                "observed": y_test.to_numpy(),
                "predicted_prob": y_prob,
            })
        )
 
    permutation_predictions = pd.concat(
        permutation_predictions,
        ignore_index=True,
    )
 
    participant_null_predictions = (
        permutation_predictions
        .groupby("participant_index", as_index=False)
        .agg(
            observed=("observed", "first"),
            predicted_prob=("predicted_prob", "mean"),
            n_predictions=("predicted_prob", "size"),
        )
    )
 
    if not (
        participant_null_predictions["n_predictions"]
        == n_outer_repeats
    ).all():
        raise RuntimeError(
            "Not every participant received the expected number of "
            "permutation OOF predictions."
        )
 
    if participant_null_predictions["observed"].nunique() >= 2:
        return roc_auc_score(
            participant_null_predictions["observed"],
            participant_null_predictions["predicted_prob"],
        )
    return np.nan
 
 
def permutation_null_aucs(
    estimator,
    X_data,
    y_data,
    n_permutations=N_PERMUTATIONS,
    n_outer_splits=PERM_OUTER_SPLITS,
    n_outer_repeats=PERM_OUTER_REPEATS,
    random_state=42,
    n_jobs=-2,
):
    """
    Empirical null distribution of participant-level aggregated OOF AUC.
 
    For each permutation:
    1. Shuffle outcome labels.
    2. Rerun nested repeated stratified CV.
    3. Average repeated OOF probabilities per participant.
    4. Calculate one participant-level AUC.
 
    This matches the construction of the observed participant-level OOF AUC.
    """
    rng = np.random.RandomState(random_state)
    y_array = y_data.to_numpy()
 
    shuffled_ys = []
    cv_seeds = []
    for _ in range(n_permutations):
        shuffled_ys.append(
            pd.Series(
                rng.permutation(y_array),
                index=y_data.index,
                name=y_data.name,
            )
        )
        cv_seeds.append(rng.randint(0, 1_000_000))
 
    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):
        results = Parallel(
            n_jobs=n_jobs,
            verbose=5,
        )(
            delayed(_fit_one_permutation)(
                estimator,
                X_data,
                shuffled_ys[perm_i],
                cv_seeds[perm_i],
                n_outer_splits,
                n_outer_repeats,
            )
            for perm_i in range(n_permutations)
        )
 
    return np.array(results, dtype=float)

def calculate_repeat_metrics(predictions_df):
    repeat_results = []

    grouping_columns = ['model', 'repeat']

    for (model_name, repeat), group in predictions_df.groupby(grouping_columns):
        y_true = group['observed'].to_numpy()
        y_prob = group['predicted_prob'].to_numpy()
        
        #apply o.5 probability threshold to get predicted classes
        y_pred = (y_prob >= 0.5).astype(int)

        repeat_results.append({
            "model": model_name,
            "repeat": repeat,
            "auc": roc_auc_score(y_true, y_prob),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=np.nan),
            "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=np.nan),
        })

    return pd.DataFrame(repeat_results)

def aggregate_participant_predictions(predictions_df):
    """
    Average repeated OOF probabilities so each participant contributes once.
    """
    participant_predictions = (
        predictions_df
        .groupby(
            ["model", "participant_index"],
            as_index=False,
        )
        .agg(
            observed=("observed", "first"),
            predicted_prob=("predicted_prob", "mean"),
            n_predictions=("predicted_prob", "size"),
        )
    )

    expected = OUTER_REPEATS
    if not (participant_predictions["n_predictions"] == expected).all():
        bad = participant_predictions[
            participant_predictions["n_predictions"] != expected
        ]
        raise RuntimeError(
            "Not every model-participant combination has the expected "
            f"{expected} OOF predictions. Problematic rows:\n{bad.head()}"
        )

    return participant_predictions

def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
 
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip resamples that happen to drop one class entirely
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
 
    boot_aucs = np.array(boot_aucs)
    lower = np.percentile(boot_aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_aucs, (1 + ci) / 2 * 100)
    return np.mean(boot_aucs), lower, upper, len(boot_aucs)

def corrected_resampled_ttest(differences, n_train, n_test, alternative="two-sided"):
    """
    Nadeau-Bengio corrected resampled t-test for matched outer-fold scores.

    This is used as a supporting analysis for B-E versus the demographic
    reference model A. It accounts approximately for dependence caused by
    overlapping outer-training samples.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[~np.isnan(differences)]

    n = len(differences)
    mean_difference = differences.mean()
    sample_variance = differences.var(ddof=1)

    correction = (1 / n) + (n_test / n_train)
    corrected_se = np.sqrt(correction * sample_variance)

    if corrected_se == 0:
        t_statistic = np.nan
        p_value = np.nan
        ci_lower = mean_difference
        ci_upper = mean_difference
    else:
        t_statistic = mean_difference / corrected_se
        df = n - 1

        if alternative == "greater":
            p_value = t.sf(t_statistic, df)
        elif alternative == "two-sided":
            p_value = 2 * t.sf(abs(t_statistic), df)
        else:
            raise ValueError(
                "alternative must be 'greater' or 'two-sided'"
            )

        critical_value = t.ppf(0.975, df)
        ci_lower = mean_difference - critical_value * corrected_se
        ci_upper = mean_difference + critical_value * corrected_se

    return {
        "mean_auc_difference_vs_A": mean_difference,
        "corrected_se_difference": corrected_se,
        "corrected_t_statistic": t_statistic,
        "corrected_p_vs_A": p_value,
        "difference_ci_lower": ci_lower,
        "difference_ci_upper": ci_upper,
        "n_valid_pairs": n,
    }

#* ── 0. Load Data ────────────────────────────────────────────
df_sleep = pd.read_csv("output/4_Prediction_Model/0_prep_data/df_features_sleep_prep_2026-07-08.csv")
df_step = pd.read_csv("output/4_Prediction_Model/0_prep_data/df_features_step_prep_2026-07-08.csv")
df_hr = pd.read_csv("output/4_Prediction_Model/0_prep_data/df_features_all_hr_prep_2026-07-08.csv")
df_demo = pd.read_csv("output/1_feature_extraction/df_anthropometric_data_2026-07-09.csv")[['bmi', 'WHtR', 'study_id']]
df_ir_metrics = pd.read_csv("data/insulin_resistance/IR_metrics.csv")[['study_id', 'matsuda_2h']].rename(columns={'matsuda_2h': 'Matsuda_Index'})



df = pd.merge(df_sleep, df_step, on='study_id', how='inner')
df = pd.merge(df, df_hr, on='study_id', how='inner')
df = pd.merge(df, df_demo, on='study_id', how='inner')
df = pd.merge(df, df_ir_metrics, on='study_id', how='left')

target = "Matsuda_Index"

demo_features = ["WHtR", "bmi"]

sleep_features = [
    "sleep_duration",
    "waso",
    "awake",
    "wakeup_seconds",
    "sleep_onset_s_adj",
    "midpoint_s_adj",
    "light_sleep_duration",
    "deep_sleep_duration",
    "rem_sleep_duration",
    "overall_sleep_score",
    "ser",
]

step_features = [
    "steps",
    "sedentary_time_h",
    "movement_time_h",
    "ratio",
    "steps_2h",
    "steps_4h",
    "steps_onset_2h",
    "steps_onset_4h",
]

hr_features = [
    "hr",
    "rmssd",
    "hr_nocturnal",
    "rmssd_nocturnal",
]
required_cols = (
    [target]
    + demo_features
    + sleep_features
    + step_features
    + hr_features
)

df_model = df[required_cols].dropna().copy()

print(f"Participants used in models: {len(df_model)} (dropped {len(df) - len(df_model)} with missing data)")

#*store config file for reproducibility

analysis_config = {
    "outer_splits": OUTER_SPLITS,
    "outer_repeats": OUTER_REPEATS,
    "inner_splits": INNER_SPLITS,
    "n_permutations": N_PERMUTATIONS,
    "permutation_outer_splits": PERM_OUTER_SPLITS,
    "permutation_outer_repeats": PERM_OUTER_REPEATS,
    "bootstrap_iterations": 2000,
    "bootstrap_ci": 0.95,
    "probability_threshold": 0.5,
    "random_state": 42,
    "C_grid": C_GRID,
    "sleep_pca_components": SLEEP_PCA_COMPONENTS,
    "step_pca_components": STEP_PCA_COMPONENTS,
    "hr_pca_components": HR_PCA_COMPONENTS,
    "candidate_cutoffs": CANDIDATE_CUTOFFS,
    "model_labels": model_labels,
    "model_order": model_order,
}

with open(
    output_dir + "analysis_config_cutoff_sensitivity.json",
    "w",
) as file:
    json.dump(analysis_config, file, indent=4)


#* ── 1. Pre-defined cutoffs based on literature (Lechner et al. (2021) ────────────────────────────────────────────

print("\nPre-specified candidate cutoffs:")
for name, c in CANDIDATE_CUTOFFS.items():
    print(f"  {name}: {c}")


#* ── 2. Check Classification Balance ────────────────────────────────────────────
print("\n")
print("-" * 20)
print("\nCheck Classification Balance")

balance_rows = []
for name, c in CANDIDATE_CUTOFFS.items():
    n_pos = int((df_model["Matsuda_Index"] <= c).sum())  # class 1 = lower insulin sensitivity (Matsuda <= cutoff)
    n_neg = int((df_model["Matsuda_Index"] > c).sum())
    n_total = n_pos + n_neg
    min_class = min(n_pos, n_neg)
 
    balance_rows.append({
        "cutoff_name": name, "cutoff_value": c, "n_total_used": n_total,
        "n_class1_lowerIS": n_pos, "n_class0_higherIS": n_neg,
        "pct_class1": round(100*n_pos/n_total, 1),
        "min_class_n": min_class, 
        "epv_at_2_predictors": round(min_class/2, 1),
        "epv_at_4_predictors": round(min_class/4, 1), 
        "epv_at_8_predictors": round(min_class/8, 1),
    })


balance_df = pd.DataFrame(balance_rows)
print("\nBalance / feasibility check across candidate cutoffs:")
print(balance_df.to_string(index=False))
balance_df.to_csv("output/4_Prediction_Model/3_Logistic_Regression/6_Final_Sensitivity/balance_check_cutoff_sensitivity.csv", index=False)


ax = sns.histplot(
    data=df_model,
    x="Matsuda_Index",
    kde=True,
    color=COLORS["blue1"],
)


cutoff_lines = {value: COLORS["red1"] for value in CANDIDATE_CUTOFFS.values()}

for cutoff_line, color in cutoff_lines.items():
    ax.axvline(
        x=cutoff_line,
        color=color,
        linestyle="--",
        linewidth=1.5,
        label=f"{cutoff_line:.1f}",
    )

ax.set(
    xlabel="Matsuda Index",
    ylabel="Count",
    title="Distribution of Matsuda Index with Candidate Cutoffs",
)

ax.legend(
    frameon=False,
    title="Cutoff values",
    title_fontsize=9,
    fontsize=9,
    loc="center left",
    bbox_to_anchor=(1.02, 0.8),
)

sns.despine(ax=ax, top=True, right=True)

ax.title.set_fontsize(11)
ax.figure.tight_layout()
# ax.figure.show()
ax.figure.savefig(
    "plots/Prediction_Model/Logistic_Regression/6_Final_Sensitivity/matsuda_distribution_cutoff_sensitivity.png",
    dpi=300,
    bbox_inches="tight",
)
df_model.to_csv("output/4_Prediction_Model/3_Logistic_Regression/6_Final_Sensitivity/matsuda_distribution_cutoff_sensitivity_data.csv", index=False)

#* ── 3. Define 5 Predictor Models with PCA fitted inside Pipeline  ────────────────────────────────────────────

models = {
    "A_demo": make_tuned_model(ColumnTransformer([
            ("demo", StandardScaler(), demo_features)
        ])),

    "B_demo_sleep": make_tuned_model(ColumnTransformer([
            ("demo", StandardScaler(), demo_features),
            ("sleep_pca", make_pca_pipeline(SLEEP_PCA_COMPONENTS), sleep_features)
        ])),
        
    "C_demo_steps": make_tuned_model(ColumnTransformer([
            ("demo", StandardScaler(), demo_features),
            ("step_pca", make_pca_pipeline(STEP_PCA_COMPONENTS), step_features)
        ])),
    
    "D_demo_hr": make_tuned_model(ColumnTransformer([
            ("demo", StandardScaler(), demo_features),
            ("hr_pca", make_pca_pipeline(HR_PCA_COMPONENTS), hr_features)
        ])),
        
    "E_all": make_tuned_model(ColumnTransformer([
            ("demo", StandardScaler(), demo_features),
            ("sleep_pca", make_pca_pipeline(SLEEP_PCA_COMPONENTS), sleep_features),
            ("step_pca", make_pca_pipeline(STEP_PCA_COMPONENTS), step_features),
            ("hr_pca", make_pca_pipeline(HR_PCA_COMPONENTS), hr_features)
        ])),
}


#* ── 4. Run all Models ────────────────────────────────────────────
print("\n")
print("-" * 20)
print("\nRun all models, store results & show dropped folds")

all_cutoff_summary_rows = []
all_fold_results = []
all_repeat_results = []
all_participant_predictions = []
all_comparison_rows = []
all_permutation_rows = []
all_null_auc_distributions = []
 
for cutoff_name, c in CANDIDATE_CUTOFFS.items():
    print(f"\n{'='*60}\nRunning cutoff: {cutoff_name} ({c})\n{'='*60}")
 
    df_cut = df_model.copy()
    df_cut["ir_class"] = (df_cut["Matsuda_Index"] <= c).astype(int)
 
    X_cut = df_cut.drop(columns=["Matsuda_Index", "ir_class"])
    y_cut = df_cut["ir_class"]

    minority_n = int(y_cut.value_counts().min())
 
    # guard: skip cutoffs where a class is too small for stratified 5-fold at all
    if minority_n < OUTER_SPLITS:
        print(f"  SKIPPED: minority class n={minority_n} is too small "
              f"for stratified {OUTER_SPLITS}-fold CV. Reporting balance only for this cutoff.")
        continue
 
    cutoff_fold_results = []
    cutoff_predictions = []
    for model_name, estimator in models.items():
        print(f"Running {model_name}")

        fold_df, prediction_df= evaluate_model(
            model = model_name, 
            model_estimator = estimator,
            X_data = X_cut, 
            y_data = y_cut, 
            cross_validation = outer_cv,
            n_outer_splits = OUTER_SPLITS
            )
        
        fold_df["cutoff_name"] = cutoff_name
        fold_df["cutoff_value"] = c

        prediction_df["cutoff_name"] = cutoff_name
        prediction_df["cutoff_value"] = c

        cutoff_fold_results.append(fold_df)
        cutoff_predictions.append(prediction_df)

        print(f"{len(fold_df)} outer folds, {len(prediction_df)} OOF prediction rows")

    cutoff_fold_results = pd.concat(cutoff_fold_results, ignore_index=True)
    cutoff_predictions = pd.concat(cutoff_predictions, ignore_index=True)

    repeat_df = calculate_repeat_metrics(cutoff_predictions)
    repeat_df["cutoff_name"] = cutoff_name
    repeat_df["cutoff_value"] = c

    participant_df = aggregate_participant_predictions(
        cutoff_predictions
    )
    participant_df["cutoff_name"] = cutoff_name
    participant_df["cutoff_value"] = c

    # Sanity checks
    expected_fold_rows = OUTER_SPLITS * OUTER_REPEATS
    expected_prediction_rows = len(X_cut) * OUTER_REPEATS

    fold_counts = cutoff_fold_results.groupby("model").size()
    prediction_counts = cutoff_predictions.groupby("model").size()
    repeat_counts = repeat_df.groupby("model").size()

    if not (fold_counts == expected_fold_rows).all():
        raise RuntimeError(
            f"Unexpected fold counts for {cutoff_name}:\n{fold_counts}"
        )

    if not (prediction_counts == expected_prediction_rows).all():
        raise RuntimeError(
            f"Unexpected prediction counts for {cutoff_name}:\n"
            f"{prediction_counts}"
        )

    if not (repeat_counts == OUTER_REPEATS).all():
        raise RuntimeError(
            f"Unexpected repeat counts for {cutoff_name}:\n"
            f"{repeat_counts}"
        )

    # Primary summary: metrics across complete CV repeats.
    for model_name, _ in models.items():
        model_repeat = repeat_df[repeat_df["model"] == model_name]
        model_participant = participant_df[participant_df["model"] == model_name]

        participant_oof_auc = roc_auc_score(model_participant["observed"],model_participant["predicted_prob"],)

        # Permutation test vs. chance, at this cutoff, for this predictor set.
        print(f"  Permutation test: {cutoff_name} / {model_name} ({N_PERMUTATIONS} permutations)...")
        null_aucs = permutation_null_aucs(models[model_name], X_cut, y_cut)
        all_null_auc_distributions.append(
            pd.DataFrame({
                "cutoff_name": cutoff_name,
                "cutoff_value": c,
                "model": model_name,
                "permutation": np.arange(1, len(null_aucs) + 1),
                "null_auc": null_aucs,
            })
        )
        null_aucs_valid = null_aucs[~np.isnan(null_aucs)]
        perm_p_value = (np.sum(null_aucs_valid >= participant_oof_auc) + 1) / (len(null_aucs_valid) + 1)
 
        all_permutation_rows.append({
            "cutoff_name": cutoff_name,
            "cutoff_value": c,
            "model": model_name,
            "observed_participant_oof_auc": participant_oof_auc,
            "null_mean_auc": np.mean(null_aucs_valid),
            "null_sd_auc": np.std(null_aucs_valid),
            "n_permutations_valid": len(null_aucs_valid),
            "n_permutations_dropped": np.isnan(null_aucs).sum(),
            "p_value": perm_p_value,
        })

        (boot_mean, boot_lower, boot_upper, n_valid_boot) = bootstrap_auc_ci(model_participant["observed"],model_participant["predicted_prob"] )

        all_cutoff_summary_rows.append({
            "cutoff_name": cutoff_name,
            "cutoff_value": c,
            "model": model_name,
            "mean_repeat_auc": model_repeat["auc"].mean(),
            "sd_repeat_auc": model_repeat["auc"].std(),
            "mean_repeat_bal_acc": (model_repeat["balanced_accuracy"].mean()),
            "sd_repeat_bal_acc": (model_repeat["balanced_accuracy"].std()),
            "mean_repeat_sensitivity": (model_repeat["sensitivity"].mean()),
            "sd_repeat_sensitivity": (model_repeat["sensitivity"].std()),
            "mean_repeat_specificity": (model_repeat["specificity"].mean()),
            "sd_repeat_specificity": (model_repeat["specificity"].std()),
            "participant_oof_auc": participant_oof_auc,
            "bootstrap_mean_auc": boot_mean,
            "ci_lower": boot_lower,
            "ci_upper": boot_upper,
            "n_valid_bootstraps": n_valid_boot,
        })

    # Supporting comparison: each lifestyle-expanded set versus Set A.
    fold_auc_wide = cutoff_fold_results.pivot(
        index=["repeat", "fold"],
        columns="model",
        values="auc",
    )

    n_total = len(X_cut)
    n_test = n_total / OUTER_SPLITS
    n_train = n_total - n_test

    for model_name in ["B_demo_sleep", "C_demo_steps", "D_demo_hr", "E_all"]:
        paired = fold_auc_wide[["A_demo", model_name]].dropna()

        differences = (paired[model_name] - paired["A_demo"])

        test_result = corrected_resampled_ttest(
            differences=differences,
            n_train=n_train,
            n_test=n_test,
            alternative="greater",
        )

        # Descriptive paired difference at the repeat level.
        repeat_wide = repeat_df.pivot(
            index="repeat",
            columns="model",
            values="auc",
        )
        repeat_differences = (repeat_wide[model_name] - repeat_wide["A_demo"])

        all_comparison_rows.append({
            "cutoff_name": cutoff_name,
            "cutoff_value": c,
            "comparison_model": model_name,
            **test_result,
            "median_repeat_auc_difference_vs_A": (
                repeat_differences.median()
            ),
            "sd_repeat_auc_difference_vs_A": (
                repeat_differences.std()
            ),
            "proportion_repeats_better_than_A": (
                repeat_differences > 0
            ).mean(),
        })

    all_fold_results.append(cutoff_fold_results)
    all_repeat_results.append(repeat_df)
    all_participant_predictions.append(participant_df)

results_by_cutoff = pd.DataFrame(all_cutoff_summary_rows)
comparison_df = pd.DataFrame(all_comparison_rows)
fold_results_all = pd.concat(all_fold_results, ignore_index=True)
repeat_results_all = pd.concat(all_repeat_results, ignore_index=True)
participant_predictions_all = pd.concat(all_participant_predictions, ignore_index=True)
null_auc_distributions_df = pd.concat(all_null_auc_distributions, ignore_index=True,)



# Flag comparisons where a meaningful share of fold-pairs were dropped
# (undefined AUC in >=1 model due to a single-class test fold) before the
# Nadeau-Bengio test ran - most likely at the more extreme cutoffs where the
# minority class is small. n_valid_pairs well below OUTER_SPLITS*OUTER_REPEATS
# means the reported test has less effective power than the full design.
max_possible_pairs = OUTER_SPLITS * OUTER_REPEATS
dropped_pairs = comparison_df.assign(pct_pairs_retained=lambda d: 100 * d["n_valid_pairs"] / max_possible_pairs)
print("\nFold-pairs retained for the corrected resampled t-test (vs. Set A):")
print(
    dropped_pairs[
        ["cutoff_name", "comparison_model", "n_valid_pairs", "pct_pairs_retained"]
    ].to_string(index=False)
)
#* ── Permutation test vs. chance (all cutoffs x all predictor sets) ────────────────────────────────────────────
#BH/FDR correction within cutoffs to match logic of Set A vs Set B-E comparison.
permutation_df = pd.DataFrame(all_permutation_rows)

permutation_df["p_value_fdr_bh"] = np.nan
permutation_df["significant_after_fdr"] = False

for cutoff, group in permutation_df.groupby("cutoff_name"):
    fdr_reject, fdr_pvals, _, _ = multipletests(
        group["p_value"], alpha=0.05, method="fdr_bh"
    )
    permutation_df.loc[group.index, "p_value_fdr_bh"] = fdr_pvals
    permutation_df.loc[group.index, "significant_after_fdr"] = fdr_reject


permutation_df["model_label"] = permutation_df["model"].map(model_labels)
 
print("\nPermutation test results (AUC vs. chance), all cutoffs x predictor sets:")
print(
    permutation_df[
        [
            "cutoff_name", "model", "observed_participant_oof_auc",
            "null_mean_auc", "p_value", "p_value_fdr_bh", "significant_after_fdr",
        ]
    ].to_string(index=False)
)
permutation_df.to_csv(
    output_dir + "permutation_test_vs_chance_cutoff_sensitivity.csv",
    index=False,
)

#* ── Set A vs Set B-E (within cutoffs) ────────────────────────────────────────────
# Holm adjustment within each cutoff across B-E versus Set A.
comparison_df["p_adj_holm"] = np.nan
comparison_df["significant_holm"] = False

for cutoff_name in comparison_df["cutoff_name"].unique():
    mask = comparison_df["cutoff_name"] == cutoff_name
    p_values = comparison_df.loc[mask, "corrected_p_vs_A"]

    reject, adjusted_p, _, _ = multipletests(
        p_values,
        method="holm",
    )

    comparison_df.loc[mask, "p_adj_holm"] = adjusted_p
    comparison_df.loc[mask, "significant_holm"] = reject

# Merge comparison statistics onto the model-level summary.
results_by_cutoff = results_by_cutoff.merge(
    comparison_df[
        [
            "cutoff_name",
            "comparison_model",
            "mean_auc_difference_vs_A",
            "difference_ci_lower",
            "difference_ci_upper",
            "corrected_p_vs_A",
            "p_adj_holm",
            "significant_holm",
            "proportion_repeats_better_than_A",
            "n_valid_pairs",
        ]
    ],
    how="left",
    left_on=["cutoff_name", "model"],
    right_on=["cutoff_name", "comparison_model"],
).drop(columns="comparison_model")

print("\nFull cutoff sensitivity summary:")
print(results_by_cutoff.to_string(index=False))



results_by_cutoff.to_csv(
    output_dir + "nested_cv_summary_cutoff_sensitivity.csv",
    index=False,
)
comparison_df.to_csv(
    output_dir + "comparisons_vs_A_cutoff_sensitivity.csv",
    index=False,
)
fold_results_all.to_csv(
    output_dir + "outer_fold_results_cutoff_sensitivity.csv",
    index=False,
)
repeat_results_all.to_csv(
    output_dir + "repeat_results_cutoff_sensitivity.csv",
    index=False,
)
participant_predictions_all.to_csv(
    output_dir + "participant_oof_predictions_cutoff_sensitivity.csv",
    index=False,
)
null_auc_distributions_df.to_csv(
    output_dir + "null_auc_distributions_cutoff_sensitivity.csv",
    index=False,
    )

#* ── 6. Comparison plot across cutoffs ────────────────────────────────────────────

results_by_cutoff["model_label"] = results_by_cutoff["model"].map(model_labels)
# Forest-style plot, one panel per cutoff actually run, so the same pattern
# (or its absence) across A-E is directly visually comparable across cutoffs.
available_cutoffs = results_by_cutoff["cutoff_name"].unique()

cutoffs_run = [
    cutoff_name
    for cutoff_name in CANDIDATE_CUTOFFS
    if cutoff_name in available_cutoffs
]

n_cutoffs = len(cutoffs_run)
n_rows = 2
n_cols = math.ceil(n_cutoffs / n_rows)

fig = plt.figure(
    figsize=(4.5 * n_cols, 4.5 * n_rows),
)

# Use twice as many GridSpec columns so an incomplete final row
# can be centered while keeping all panels equally sized.
gs = fig.add_gridspec(
    nrows=n_rows,
    ncols=2 * n_cols,
    hspace=0.4,
    wspace=0.45,
)

axes = []

# Number of panels in each row
n_top = min(n_cols, n_cutoffs)
n_bottom = n_cutoffs - n_top

# First row
for i in range(n_top):
    axes.append(
        fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    )

# Second row, centered when it is not full
if n_bottom > 0:
    start_col = n_cols - n_bottom

    for i in range(n_bottom):
        axes.append(
            fig.add_subplot(
                gs[1, start_col + 2 * i:start_col + 2 * i + 2]
            )
        )

for ax, cutoff_name in zip(axes, cutoffs_run):
    sub = results_by_cutoff[
        results_by_cutoff["cutoff_name"] == cutoff_name
    ].copy()

    # Enforce the same model order in every panel
    sub["model_label"] = pd.Categorical(
        sub["model_label"],
        categories=model_colors.keys(),
        ordered=True,
    )

    sub = (
        sub
        .dropna(subset=["model_label"])
        .sort_values("model_label")
        .reset_index(drop=True)
    )

    y_pos = np.arange(len(sub))

    # Plot one model at a time so each gets its own color
    for i, row in sub.iterrows():
        color = model_colors[row["model_label"]]

        ax.errorbar(
            row["participant_oof_auc"],
            y_pos[i],
            xerr=[[row["participant_oof_auc"] - row["ci_lower"]], [row["ci_upper"] - row["participant_oof_auc"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            elinewidth=1.3,
            capthick=1.3,
            markersize=6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["model_label"])

    ax.axvline(
        0.5,
        color=COLORS["red1"],
        linestyle="--",
        alpha=0.9,
    )

    ax.set_xlabel("Aggregated oof AUC (95% bootstrap CI)")
    ax.set_title(
        f"Cutoff {CANDIDATE_CUTOFFS[cutoff_name]:.1f}",
        fontsize=10,
    )
    ax.set_xlim(0.2, 1.0)

    sns.despine(ax=ax, top=True, right=True)



legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color=model_colors[label],
        markerfacecolor=model_colors[label],
        markersize=6,
        linestyle="None",
        label=label,
    )
    for label in model_colors.keys()
]

legend_handles.append(
    Line2D(
        [0], [0],
        color=COLORS["red1"],
        linestyle="--",
        label="Chance level (AUC = 0.5)",
    )
)

fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.89, 0.5),
    frameon=False,
)

fig.suptitle(
    "Participant-level out-of-fold AUC across candidate cutoffs",
    fontsize=11,
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])

plt.savefig(
    "plots/Prediction_Model/Logistic_Regression/6_Final_Sensitivity/bootstrap_auc_ci_cutoff_sensitivity.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


#* ── 7. Plots ────────────────────────────────────────────
cutoff_labels = {
    "cutoff_2.5": "2.5",
    "cutoff_3.5": "3.5",
    "cutoff_4.0": "4.0 (primary)",
    "cutoff_4.3": "4.3",
    "cutoff_6.4": "6.4",
}
cutoff_order = list(CANDIDATE_CUTOFFS.keys())
cutoffs_present = [
    cutoff_name
    for cutoff_name in cutoff_order
    if cutoff_name in results_by_cutoff["cutoff_name"].unique()
]


#* ── Figure: grouped bar chart with 95% CI error bars, one group per predictor set ────────────────────────────────────────────
#merge permutation results with resulty by cutoff
results_by_cutoff = results_by_cutoff.merge(
    permutation_df[["cutoff_name", "model", "p_value_fdr_bh", "significant_after_fdr", "null_mean_auc"]],
    how="left",
    on=["cutoff_name", "model"],
)
# A model is marked robust only if it significantly outperforms Set A at
# every cutoff that was actually analysed. Set A itself is not tested.
significance_table = (
    results_by_cutoff[
        results_by_cutoff["model"] != "A_demo"
    ]
    .groupby("model")["significant_holm"]
    .apply(lambda values: bool(values.all()))
)


n_models = len(model_order)
n_cutoffs = len(cutoffs_present)

model_base_colors = {
    "A_demo": COLORS["blue1"],
    "B_demo_sleep": COLORS["orange1"],
    "C_demo_steps": COLORS["green1"],
    "D_demo_hr": COLORS["violet1"],
    "E_all": COLORS["teal1"],
}
# ── One base hue per predictor set; shades within that hue encode cutoff ──
# (lightest = lowest cutoff, most saturated/darkest = highest cutoff).

# generate n_cutoffs shades per model, light -> base color (darkest = highest cutoff)
model_shades = {
    model_name: sns.light_palette(
        model_base_colors[model_name],
        n_colors=n_cutoffs + 1,
    )[1:]
    for model_name in model_order
}


plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(11, 6))

bar_width = 0.18
group_gap = 0.05
x_base = np.arange(n_models)

for i, cutoff in enumerate(cutoffs_present):
    sub = results_by_cutoff[results_by_cutoff["cutoff_name"] == cutoff].set_index("model").reindex(model_order)
    offset = (i - (n_cutoffs - 1) / 2) * (bar_width + group_gap / n_cutoffs)
    x_pos = x_base + offset

    yerr = np.array([
        sub["participant_oof_auc"] - sub["ci_lower"],
        sub["ci_upper"] - sub["participant_oof_auc"]
    ])

    bar_colors = [model_shades[m][i] for m in model_order]

    ax.bar(
        x_pos,
        sub["participant_oof_auc"],
        width=bar_width,
        yerr=yerr, 
        capsize=3,
        color=bar_colors,
        edgecolor="white", 
        linewidth=0.5,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"}
    )

    # Stars mark Holm-adjusted significance versus Set A.
    for x, model_name, sig_vs_a, sig_vs_null, upper in zip(
        x_pos,
        model_order,
        sub["significant_holm"],
        sub['significant_after_fdr'],
        sub["ci_upper"],
    ):
        markers =[]
        if model_name != "A_demo" and bool(sig_vs_a):
            markers.append("*")
        if bool(sig_vs_null):
            markers.append("▲")
        if markers:
            ax.text(
                x,
                upper + 0.015,
                "".join(markers),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )

ax.axhline(0.5, color=COLORS["red1"], linestyle="--", linewidth=1, alpha=0.9, label="Chance level (AUC = 0.5)")

ax.set_xticks(x_base)
ax.set_xticklabels([model_labels[m] for m in model_order], fontsize=10)
ax.set_ylabel("Aggregated oof AUC (95% bootstrap CI)", fontsize=10)
ax.set_ylim(0.25, 0.95)
ax.set_title("Classification performance across predictor sets and cutoffs\n"
             "(* = significant vs. Set A, Holm adjustment; ▲ = significant vs. permutation null, BH/FDR adjusted)", fontsize=11)

# highlight robust sets with a subtle background band
for i, m in enumerate(model_order):
    if significance_table.get(m, False):
        ax.axvspan(
            i - 0.45, 
            i + 0.45, 
            color=COLORS["grey1"], 
            alpha=0.15, 
            zorder=0)

# Legend part 1: chance-level reference line
line_handle, line_label = ax.get_legend_handles_labels()
chance_legend = ax.legend(line_handle, 
                          line_label, 
                          loc="upper left",
                          bbox_to_anchor=(1.01, 1.0), 
                          frameon=False, 
                          fontsize=9)
ax.add_artist(chance_legend)

# Legend part 2: cutoff shade key, drawn as a small horizontal swatch strip
# in neutral gray so it reads as "lighter -> darker", independent of any one
# predictor set's hue.

gray_shades = sns.light_palette("#555555", n_colors=n_cutoffs + 1)[1:]

shade_handles = [Patch(facecolor=gray_shades[i], edgecolor="white",
                        label=cutoff_labels[cutoffs_present[i]])
                  for i in range(n_cutoffs)]
shade_legend = ax.legend(
    handles=shade_handles, 
    title="Cutoff\n(lighter = lower)",
    loc="upper left", 
    bbox_to_anchor=(1.01, 0.9), 
    frameon=False, 
    fontsize=9,
    title_fontsize=9)

robust_names = [model_labels[m].split(" — ")[0] for m in model_order if significance_table.get(m, False)]
if robust_names:
    ax.text(0.01, 0.97, f"Shaded = robust across all {n_cutoffs} cutoffs (Set {', '.join(robust_names)})",
            transform=ax.transAxes, fontsize=8.5, va="top", color=COLORS["grey1"], style="italic")

plt.tight_layout()
plt.savefig("plots/Prediction_Model/Logistic_Regression/6_Final_Sensitivity/mean_auc_cutoff_sensitivity.png", dpi=300,
            bbox_inches="tight", bbox_extra_artists=(chance_legend, shade_legend))
plt.show()


print("END OF SCRIPT")

