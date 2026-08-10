import numpy as np
import pandas as pd
from scipy.stats import t

from joblib import Parallel, delayed, parallel_config

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_curve   
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests

from scripts.plot_helper.colors import COLORS


import seaborn as sns
import matplotlib.pyplot as plt

import json

#! Keep Baseline only for development purposes but exclude from plots/tables in final analysis


#* ── Constants ────────────────────────────────────────────
OUTER_SPLITS = 5
OUTER_REPEATS = 50
INNER_SPLITS = 4

#Permutation Testing
N_PERMUTATIONS = 1000 #builds empirical null distribution
PERM_OUTER_SPLITS = 5 # controls how many repeated outer-CV partitions are pooled within single permutation draw
PERM_OUTER_REPEATS = 5 #smaller than OUTER_REPEATS to reduce computation time, but still enough to get a sense of the null distribution; increase if possible


outer_cv = RepeatedStratifiedKFold(
    n_splits=OUTER_SPLITS,
    n_repeats=OUTER_REPEATS,
    random_state=42
)
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=42)



C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]

SLEEP_PCA_COMPONENTS = 3
STEP_PCA_COMPONENTS = 3
HR_PCA_COMPONENTS = 2

label_map = {
    "Baseline_prior": "Baseline",
    "A_demo": "Set A",
    "B_demo_sleep": "Set B",
    "C_demo_steps": "Set C",
    "D_demo_hr": "Set D",
    "E_all": "Set E",
}

model_colors = {
    # "Baseline": COLORS["grey1"],
    "Set A": COLORS["blue1"],
    "Set B": COLORS["orange1"],
    "Set C": COLORS["green1"],
    "Set D": COLORS["violet1"],
    "Set E": COLORS["teal1"],
}
model_colors_2 = {
    # "Baseline": COLORS["grey2"],
    "Set A": COLORS["blue2"],
    "Set B": COLORS["orange2"],
    "Set C": COLORS["green2"],
    "Set D": COLORS["violet2"],
    "Set E": COLORS["teal2"],
    
}

model_order = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']  # order for plots


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


def calculate_repeat_metrics(predictions_df):
    repeat_results = []

    grouping_columns = ['model', 'repeat']

    for (model_name, repeat), group in predictions_df.groupby(grouping_columns):
        y_true = group['observed'].to_numpy()
        y_prob = group['predicted_prob'].to_numpy()
        
        #apply 0.5 probability threshold to get predicted classes
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

# Repeated-CV mean/SD (Section 6) reflects variability across fold assignments,
# but doesn't directly give a confidence interval on AUC itself. With only
# ~15 events, a point-estimate AUC alone overstates precision — bootstrap
# resampling of the out-of-fold (observed, predicted_prob) pairs gives an
# interval that's standard to report for small-sample classification work.
#
# NOTE: this resamples the already-collected out-of-fold predictions
# (pooled across all repeats/folds for each model), not the raw data or CV
# procedure itself. It quantifies uncertainty in the AUC estimate given the
# predictions obtained, and is a complement to, not a replacement for, the
# repeated-CV SD in Section 6.
 
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
 
    This is the expensive, embarrassingly-parallel unit of work factored out
    of permutation_null_aucs so it can be dispatched via joblib. It contains
    no randomness of its own beyond what perm_cv derives from cv_seed, so
    farming it out to worker processes cannot change results.
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
 
    Parallelized across permutations with joblib. All randomness (the label
    shuffle and the per-permutation CV fold seed) is drawn sequentially,
    in the same order as the original serial implementation, *before* any
    parallel dispatch happens. This guarantees the exact same sequence of
    random numbers is consumed regardless of n_jobs or worker scheduling,
    This preserves the same shuffled labels and CV seeds regardless of
    worker scheduling, so the parallel and serial implementations should
    produce equivalent results up to numerical precision. Only the
    (expensive, randomness-free) model fitting is farmed out in parallel.
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

def corrected_resampled_ttest(differences, n_train, n_test, alternative="two-sided"):
    """
    Nadeau-Bengio corrected resampled t-test for matched outer-fold scores.
 
    Standard paired t-tests assume independent observations, which repeated
    k-fold CV violates (outer-training folds overlap across repeats). This
    correction inflates the standard error by (1/n + n_test/n_train) to
    approximately account for that dependence, and is applied at the
    fold level (matched by repeat and fold), which is the unit the
    correction was derived for.
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
            raise ValueError("alternative must be 'greater' or 'two-sided'")
 
        critical_value = t.ppf(0.975, df)
        ci_lower = mean_difference - critical_value * corrected_se
        ci_upper = mean_difference + critical_value * corrected_se
 
    return {
        "fold_mean_auc_difference_vs_A": mean_difference,
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
df_ir_metrics['lower_IS'] = (df_ir_metrics['Matsuda_Index'] <= 4.0).astype(int)


df = pd.merge(df_sleep, df_step, on='study_id', how='inner')
df = pd.merge(df, df_hr, on='study_id', how='inner')
df = pd.merge(df, df_demo, on='study_id', how='inner')
df = pd.merge(df, df_ir_metrics, on='study_id', how='left')

num_features = len(df.columns)-1
num_participants = len(df)

print(f"Number of features: {num_features}, Number of participants: {num_participants}")


#* ── 0.1 Check Classification Balance ────────────────────────────────────────────
print("\n")
print("-" * 20)
print("\nCheck Classification Balance")

df_check = df.copy()
df_check = df_check.dropna()
n_IR = df_check['lower_IS'].sum()
n_IS = (df_check['lower_IS'] == 0).sum()

print("Number of patients with Matsuda Index >4:", n_IS, f"({n_IS/num_participants*100:.2f}%)")
print("Number of patients with Matsuda Index <=4:", n_IR, f"({n_IR/num_participants*100:.2f}%)")

#save numbers and percentages to csv
matsuda_classification = pd.DataFrame({
    "Matsuda_Index": [">4", "<=4"],
    "Count": [n_IS, n_IR],
    "Percentage": [n_IS/num_participants*100, n_IR/num_participants*100]
})
matsuda_classification.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/matsuda_classification.csv", index=False)


#participants close to cut off
for margin in [0.25, 0.5, 1.0]:
    close = df_check["Matsuda_Index"].between(
        4.0 - margin,
        4.0 + margin
    ).sum()

    print(f"Within ±{margin}: {close} participants")


ax = sns.histplot(
    data=df_check,
    x="Matsuda_Index",
    kde=True,
    color=COLORS["blue1"],
)

ax.axvline(
    x=4.0,
    color=COLORS["red1"],
    linestyle="--",
    linewidth=1.5,
    label=f"{4.0:.1f}",
)

ax.set(
    xlabel="Matsuda Index",
    ylabel="Count",
    title="Distribution of Matsuda Index with Candidate Cutoff",
)

ax.legend(
    frameon=False,
    title="Cutoff value",
    title_fontsize=9,
    fontsize=9,
    # loc="center left",
    # bbox_to_anchor=(1.02, 0.8),
)

sns.despine(ax=ax, top=True, right=True)
ax.title.set_fontsize(11)
ax.figure.tight_layout()
ax.figure.savefig(
    "plots/Prediction_Model/Logistic_Regression/5_Final/matsuda_index_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
# ax.figure.show()

df_check.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/matsuda_index_distribution_data.csv", index=False)

#* ── 1. Define Outcome and Feature Groups ────────────────────────────────────────────
target = "lower_IS"


# Decide based on literature
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

#* ── 2. Prepare Dataset ────────────────────────────────────────────
print("\n")
print("-" * 20)
print("\nPrepare Dataset")
required_cols = (
    [target]
    + demo_features
    + sleep_features
    + step_features
    + hr_features
)

df_model = df[required_cols].dropna().copy()

X = df_model.drop(columns=[target])
y = df_model[target]

print(f"Participants used in models: {len(df_model)} (dropped {len(df) - len(df_model)} with missing data)")

#* ── 3. Store Config File  ────────────────────────────────────────────
analysis_config = {
    "matsuda_cutoff": 4.0,
    "classification_rule": "lower_IS = Matsuda_Index <= 4.0",
    "probability_threshold": 0.5,

    "outer_splits": OUTER_SPLITS,
    "outer_repeats": OUTER_REPEATS,
    "inner_splits": INNER_SPLITS,

    "n_permutations": N_PERMUTATIONS,
    "permutation_outer_splits": PERM_OUTER_SPLITS,
    "permutation_outer_repeats": PERM_OUTER_REPEATS,

    "bootstrap_iterations": 2000,
    "bootstrap_ci": 0.95,

    "random_state": 42,
    "C_grid": C_GRID,

    "sleep_pca_components": SLEEP_PCA_COMPONENTS,
    "step_pca_components": STEP_PCA_COMPONENTS,
    "hr_pca_components": HR_PCA_COMPONENTS,

    "demo_features": demo_features,
    "sleep_features": sleep_features,
    "step_features": step_features,
    "hr_features": hr_features,

    "model_order": model_order,
    "label_map": label_map,
}

with open(
    "output/4_Prediction_Model/3_Logistic_Regression/5_Final/"
    "analysis_config.json",
    "w",
) as file:
    json.dump(analysis_config, file, indent=4)


#* ── 4. Define 5 Predictor Models with PCA fitted inside Pipeline  ────────────────────────────────────────────

models = {
    "Baseline_prior": Pipeline([
    ("model", DummyClassifier(strategy="prior", random_state=42))
        ]),

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


#* ── 5. Run all Models ────────────────────────────────────────────
print("\n")
print("-" * 20)
print("\nRun all models, store results & show dropped folds")
all_results = []
all_predictions = []

for model_name, estimator in models.items():
    print(f"Running {model_name}")

    result_df, pred_df= evaluate_model(
        model = model_name, 
        model_estimator = estimator,
        X_data = X, 
        y_data = y, 
        cross_validation = outer_cv,
        n_outer_splits = OUTER_SPLITS
        )
    
    print(
        f"{model_name}: "
        f"{len(result_df)} folds, "
        f"{len(pred_df)} predictions"
    )

    all_results.append(result_df)
    all_predictions.append(pred_df)
 
all_results = pd.concat(all_results, ignore_index=True)
all_predictions = pd.concat(all_predictions, ignore_index=True)

print("\nResults per model:")
print(all_results.groupby("model").size())

print("\nPredictions per model:")
print(all_predictions.groupby("model").size())

print("\nRepeats per model:")
print(all_predictions.groupby("model")["repeat"].nunique())

all_results.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_all_results.csv", index=False)
all_predictions.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_all_predictions.csv", index=False)
# report how many folds had to be dropped for AUC due to single-class test folds
n_nan_auc = all_results["auc"].isna().sum()
print(f"\nFolds with undefined AUC (single-class test fold): {n_nan_auc} "
      f"out of {len(all_results)} ({100*n_nan_auc/len(all_results):.1f}%)")


repeat_results = calculate_repeat_metrics(all_predictions)
print("\nEvery model should have 50 rows")
print(repeat_results.groupby("model").size())
repeat_results.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_repeat_results.csv", index=False)


#* ── 6. Create Summary Table ────────────────────────────────────────────
print("\n")
print("-" * 20)
print("\nSummary Table")
summary = (
    repeat_results
    .groupby("model")
    .agg(
        mean_auc=("auc", "mean"),
        sd_auc=("auc", "std"),
        mean_bal_acc=("balanced_accuracy", "mean"),
        sd_bal_acc=("balanced_accuracy", "std"),
        mean_sensitivity=("sensitivity", "mean"),
        sd_sensitivity=("sensitivity", "std"),
        mean_specificity=("specificity", "mean"),
        sd_specificity=("specificity", "std"),
        n_repeats=("repeat", "count"),
        n_valid_auc_folds=("auc", "count")
    )
    .reset_index()
    .sort_values("mean_auc", ascending=False)
)

print(summary)
summary.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_repeat_results_summary.csv", index=False)

#* ── 6.5 Aggregate repeated predictions per participant ────────────────────────────────────────────
participant_predictions = (
    all_predictions
    .groupby(["model", "participant_index"], as_index=False)
    .agg(
        observed=("observed", "first"),
        prediction_prob =("predicted_prob", "mean"),
        n_predictions = ("predicted_prob", "size")
    )
)

print("\nEvery participant should have 50 predictions per model")
print(participant_predictions.groupby(["model", "participant_index"]).size())

#* ── 7. Plots ────────────────────────────────────────────
repeat_results["model_label"] = repeat_results["model"].map(label_map)
all_results["model_label"] = all_results["model"].map(label_map)
all_predictions["model_label"] = all_predictions["model"].map(label_map)
participant_predictions["model_label"] = participant_predictions["model"].map(label_map)
summary['model_label'] = summary['model'].map(label_map)

#drop Baseline from plots and tables
repeat_results = repeat_results[repeat_results["model_label"] != "Baseline"]
all_results = all_results[all_results["model_label"] != "Baseline"]
all_predictions = all_predictions[all_predictions["model_label"] != "Baseline"]
participant_predictions = participant_predictions[participant_predictions["model_label"] != "Baseline"]
summary = summary[summary["model_label"] != "Baseline"]


#*Bar plot of cross-validated AUC
plt.figure(figsize=(8, 5))
 
sns.barplot(
    data=repeat_results,
    x="model_label",
    y="auc",
    errorbar="sd",
    palette=[
        COLORS.get("blue1"),
        COLORS.get("orange1"),
        COLORS.get("green1"),
        COLORS.get("violet1"),
        COLORS.get("teal1"),]
)
 
plt.axhline(0.5, color=COLORS.get("red1"), linestyle="--", label="Chance level (AUC = 0.5)")
plt.ylabel("AUC")
plt.xlabel("Predictor set")
plt.title("Mean AUC across repeated stratified cross-validation", fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.show()


#*Boxplot of AUC across CV folds
plt.figure(figsize=(8, 5))
 
sns.boxplot(
    data=repeat_results,
    x="model_label",
    y="auc",
    order=model_order,
    palette=model_colors,
)
 
sns.stripplot(
    data=repeat_results,
    x="model_label",
    y="auc",
    order=model_order,
    palette=model_colors_2,
    size=3,
    jitter=True
)
sns.despine(top=True, right=True)
plt.axhline(0.5, color=COLORS.get("red1"), linestyle="--", label="Chance level (AUC = 0.5)",)
plt.ylabel("AUC")
plt.xlabel("Predictor set")
plt.title("AUC across repeated stratified CV runs", fontsize=11)
plt.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
)
plt.tight_layout()
plt.savefig("plots/Prediction_Model/Logistic_Regression/5_Final/AUC_across_repeats.png", dpi=300)
plt.show()
repeat_results.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/AUC_across_repeats_data.csv", index=False)


#*Predicted probability distribution by observed class

for model_name in [ "Set A", "Set B", "Set C", "Set D", "Set E"]:
    plot_df = participant_predictions[participant_predictions["model_label"] == model_name]

    plot_df["observed_label"] = plot_df["observed"].map({
        0.0: "higher IS",
        1.0: "lower IS",
    })
 
    plt.figure(figsize=(6, 5))
 
    ax = sns.histplot(
        data=plot_df,
        x="prediction_prob",
        hue="observed_label",
        bins=20,
        kde=True,
        hue_order=["higher IS", "lower IS"],
        palette={
            "higher IS": COLORS.get("grey1"),
            "lower IS": model_colors[model_name]
            },
        multiple="layer"
        
    )

    sns.despine(ax=ax, top=True, right=True)
    ax.axvline(
        0.5,
        color=COLORS["red1"],
        linestyle="--",
        alpha=0.9,
    )

    ax.set_xlabel("Predicted probability of lower IS")
    ax.set_title(
        f"Aggregated out-of-fold probability by observed class ({model_name})",
        fontsize=11,
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.set_title("Observed class")

    plt.tight_layout()
    plt.savefig(f"plots/Prediction_Model/Logistic_Regression/5_Final/predicted_prob_histogram_by_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
participant_predictions.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/participant_predictions_data.csv", index=False)


#*ROC Curves (pooled out-of-fold, participant-level)
print("\n")
print("-" * 20)
print("\nROC Curves (pooled out-of-fold predictions, participant-level)")
 
plt.figure(figsize=(6.5, 6.5))
 
for model_name in model_order:
    plot_df = participant_predictions[participant_predictions["model_label"] == model_name]
    fpr, tpr, _ = roc_curve(plot_df["observed"], plot_df["prediction_prob"])
    model_auc = roc_auc_score(plot_df["observed"], plot_df["prediction_prob"])
 
    plt.plot(
        fpr, tpr,
        color=model_colors[model_name],
        linewidth=2,
        label=f"{model_name} (AUC = {model_auc:.2f})",
    )
 
plt.plot([0, 1], [0, 1], color=COLORS.get("grey1"), linestyle="--", linewidth=1, label="Chance level")
 
sns.despine(top=True, right=True)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves per predictor set\n(pooled out-of-fold predictions, participant-level)", fontsize=11)
plt.legend(frameon=False, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/Prediction_Model/Logistic_Regression/5_Final/roc_curves_pooled.png", dpi=300, bbox_inches="tight")

plt.show()

#*Confusion Matrices (pooled out-of-fold, participant-level, threshold = 0.5) 
print("\n")
print("-" * 20)
print("\nConfusion Matrices (pooled out-of-fold predictions, participant-level, threshold = 0.5)")

fig = plt.figure(figsize=(12, 7.5))

gs = fig.add_gridspec(
    nrows=2,
    ncols=6,
    hspace=0.45,
    wspace=0.55,
)

# Three equally sized plots in the first row
axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),

    # Two equally sized, centered plots in the second row
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]
 
for ax, model_name in zip(axes, model_order):
    plot_df = participant_predictions[participant_predictions["model_label"] == model_name]
    y_true = plot_df["observed"].to_numpy()
    y_pred = (plot_df["prediction_prob"].to_numpy() >= 0.5).astype(int)
 
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: high IS", "Actual: low IS"],
        columns=["Pred: high IS", "Pred: low IS"],
    )
 
    print(f"\n{model_name} (participant-level, pooled OOF, threshold=0.5):")
    print(cm_df)
 
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(model_colors[model_name], as_cmap=True),
        cbar=False,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(model_name, fontsize=10)
    ax.set_ylabel("")
    ax.set_xlabel("")


fig.suptitle(
    "Confusion matrices per predictor set\n(pooled out-of-fold predictions, participant-level, threshold = 0.5)",
    fontsize=11, y=0.99
)
plt.tight_layout()
plt.savefig("plots/Prediction_Model/Logistic_Regression/5_Final/confusion_matrices_pooled.png", dpi=300, bbox_inches="tight")

plt.show()

#*--8. Comparison to Set A ----------------------------
print("\n")
print("-" * 20)
print("\nComparison to Set A")
reference_model = "A_demo"

comparison_rows = []

for model_name in models.keys():

    if model_name == reference_model:
        continue
    elif model_name == "Baseline_prior":
        continue

    paired = (
        repeat_results[
            repeat_results["model"].isin(
                [reference_model, model_name]
            )
        ]
        .pivot(
            index="repeat",
            columns="model",
            values="auc"
        )
        .dropna()
    )

    differences = (
        paired[model_name]
        - paired[reference_model]
    )

    comparison_rows.append({
        "comparison": (
            f"{model_name} minus {reference_model}"
        ),
        "mean_auc_difference": differences.mean(),
        "median_auc_difference": differences.median(),
        "sd_auc_difference": differences.std(),
        "proportion_repeats_better": (
            differences > 0
        ).mean()
    })

comparison_summary = pd.DataFrame(
    comparison_rows
)

# Formal test: fold-level (repeat, fold) matched AUC differences, corrected
# for the dependence induced by overlapping outer-training folds (Nadeau &
# Bengio, 2003). This is the fold-level unit the correction was derived for
# - the repeat-level descriptive stats above stay as a complementary,
# easier-to-interpret summary, not a substitute for a formal test.
fold_auc_wide = (
    all_results
    .pivot(index=["repeat", "fold"], columns="model", values="auc")
)
 
n_total = len(X)
n_test = n_total / OUTER_SPLITS
n_train = n_total - n_test
 
formal_test_rows = []
for model_name in models.keys():
    if model_name in (reference_model, "Baseline_prior"):
        continue
 
    paired = fold_auc_wide[[reference_model, model_name]].dropna()
    differences = paired[model_name] - paired[reference_model]
 
    test_result = corrected_resampled_ttest(
        differences,
        n_train=n_train,
        n_test=n_test,
        alternative="greater",
    )
    test_result["comparison"] = f"{model_name} minus {reference_model}"
    formal_test_rows.append(test_result)
 
formal_test_df = pd.DataFrame(formal_test_rows)
 
# Holm correction across the B/C/D/E vs. A family (one correction, since
# this script runs a single cutoff).
reject, adjusted_p, _, _ = multipletests(
    formal_test_df["corrected_p_vs_A"], method="holm"
)
formal_test_df["p_adj_holm"] = adjusted_p
formal_test_df["significant_holm"] = reject
 
comparison_summary = comparison_summary.merge(formal_test_df, on="comparison", how="left")

print("\nFold-pairs retained for the corrected resampled t-test (vs. Set A):")
print(
    comparison_summary[["comparison", "n_valid_pairs"]].assign(
        max_possible_pairs=OUTER_SPLITS * OUTER_REPEATS
    ).to_string(index=False)
)

print(comparison_summary)
comparison_summary.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_results_comparisons_to_A.csv", index=False)


#*Bootstrap CI on AUC
print("\n")
print("-" * 20)
print("\nBootstrap CI")

 
bootstrap_results = []
 
for model_name in models.keys():
    if model_name == "Baseline_prior":
        continue
    model_preds = participant_predictions[participant_predictions["model"] == model_name]
    boot_mean, boot_lower, boot_upper, n_valid_boot = bootstrap_auc_ci(
        model_preds["observed"], 
        model_preds["prediction_prob"]
    )
    point_auc = roc_auc_score(model_preds["observed"], model_preds["prediction_prob"])

    bootstrap_results.append({
        "model": model_name,
        "participant_oof_auc": point_auc,
        "boot_mean_auc": boot_mean,
        "ci_lower": boot_lower,
        "ci_upper": boot_upper,
        "n_valid_bootstraps": n_valid_boot
    })
 
bootstrap_df = pd.DataFrame(bootstrap_results).sort_values("boot_mean_auc", ascending=False)

bootstrap_df["model_label"] = bootstrap_df["model"].map(label_map)
print("\nBootstrap 95% CI on AUC (pooled out-of-fold predictions):")
print(bootstrap_df.to_string(index=False))
bootstrap_df.to_csv("output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_results_bootstrap_ci.csv", index=False)
 
# LOOK OUT FOR: wide intervals (e.g. AUC 0.60 [0.38, 0.80]) are the expected,
# honest consequence of n_events ~ 15 — report as-is rather than treating
# width as an error. If ci_lower is below 0.5 for a model, that model's
# discriminative ability is not distinguishable from chance at this sample size.
 
#*Forest-style plot of bootstrap AUC CIs across predictor sets
plt.figure(figsize=(10, 5))
 
y_pos = np.arange(len(bootstrap_df))
#order bootstrap_df by model_order for plotting
bootstrap_df = bootstrap_df.set_index("model_label").loc[model_order].reset_index()
for i, (_, row) in enumerate(bootstrap_df.iterrows()):
    c = model_colors[row["model_label"]]

    plt.errorbar(
        row["participant_oof_auc"],
        y_pos[i],
        xerr=[[row["participant_oof_auc"] - row["ci_lower"]],
              [row["ci_upper"] - row["participant_oof_auc"]]],
        fmt="o",
        color=c,
        ecolor=c,
        capsize=4,
    )


sns.despine(top=True, right=True)
plt.yticks(y_pos, bootstrap_df["model_label"])
plt.axvline(0.5, color=COLORS.get("red1"), linestyle="--", label="Chance level (AUC = 0.5)")
plt.xlabel("Aggregated OOF AUC with 95% bootstrap CI")
plt.title("Participant-level bootstrap confidence intervals for aggregated out-of-fold AUC", fontsize=11)
plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plots/Prediction_Model/Logistic_Regression/5_Final/bootstrap_ci.png", dpi=300, bbox_inches="tight")
plt.show()

#*-- 9. Permutation Test (model AUC vs. chance)-----------------------
print("\n")
print("-" * 20)
print("\nPermutation Test (AUC vs. chance)")
 
permutation_results = []
null_auc_distributions = {}
 
for model_name, estimator in models.items():
    if model_name == "Baseline_prior":
        continue
 
    print(f"\nRunning permutation test for {model_name} ({N_PERMUTATIONS} permutations)...")
 
    observed_auc = bootstrap_df.loc[
        bootstrap_df["model"] == model_name, "participant_oof_auc"
    ].iloc[0]
 
    null_aucs = permutation_null_aucs(estimator, X, y, n_jobs=-2)
    null_auc_distributions[model_name] = null_aucs
 
    null_aucs_valid = null_aucs[~np.isnan(null_aucs)]
    n_dropped = np.isnan(null_aucs).sum()
 
    # one-sided: proportion of null draws at least as extreme as observed,
    # with the standard +1/+1 correction so p is never exactly 0
    p_value = (np.sum(null_aucs_valid >= observed_auc) + 1) / (len(null_aucs_valid) + 1)
 
    permutation_results.append({
        "model": model_name,
        "observed_participant_oof_auc": observed_auc,
        "null_mean_auc": np.mean(null_aucs_valid),
        "null_sd_auc": np.std(null_aucs_valid),
        "n_permutations_valid": len(null_aucs_valid),
        "n_permutations_dropped": n_dropped,
        "p_value": p_value,
    })
 
permutation_df = pd.DataFrame(permutation_results)
permutation_df["model_label"] = permutation_df["model"].map(label_map)

#Benjamin-Hochberg FDR correction for multiple comparisons
fdr_reject, fdr_pvals, _, _ = multipletests(
    permutation_df["p_value"], alpha=0.05, method="fdr_bh"
)
permutation_df["p_value_fdr_bh"] = fdr_pvals
permutation_df["significant_after_fdr"] = fdr_reject

print("\nPermutation test results (AUC vs. chance):")
print(permutation_df.to_string(index=False))
permutation_df.to_csv(
    "output/4_Prediction_Model/3_Logistic_Regression/5_Final/RLRM_results_permutation_test.csv",
    index=False,
)
null_auc_distributions_df = pd.DataFrame.from_dict(null_auc_distributions, orient="index").transpose()
null_auc_distributions_df.to_csv(
    "output/4_Prediction_Model/3_Logistic_Regression/5_Final/null_auc_distributions.csv",
    index=False,
)
 
# LOOK OUT FOR: with this sample size the null distribution is not
# guaranteed to be tightly centered at 0.5 - report null_mean_auc alongside
# the p-value rather than assuming 0.5 is the null's center.
 
#* Histogram of null distribution per predictor set, observed AUC overlaid

fig = plt.figure(figsize=(12, 7.5))

gs = fig.add_gridspec(
    nrows=2,
    ncols=6,
    hspace=0.5,
    wspace=0.6,
)

# 3 plots top row, 2 centered bottom row
axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]

 
for ax, model_name_label in zip(axes, model_order):
    model_key = [k for k, v in label_map.items() if v == model_name_label][0]
    null_aucs_valid = null_auc_distributions[model_key][~np.isnan(null_auc_distributions[model_key])]
    row = permutation_df[permutation_df["model"] == model_key].iloc[0]
 
    ax.hist(
        null_aucs_valid,
        bins=25,
        color=model_colors[model_name_label],
        alpha=0.6,
        edgecolor="white",
    )
    ax.axvline(
        row["observed_participant_oof_auc"],
        color=COLORS.get("red1"),
        linestyle="-",
        linewidth=2,
        label=f"Observed AUC = {row['observed_participant_oof_auc']:.2f}",
    )
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_title(
        f"{model_name_label}\n"
        f"p = {row['p_value']:.3f} (raw), {row['p_value_fdr_bh']:.3f} (FDR)",
        fontsize=9,
    )
    ax.set_xlabel("Null AUC")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

# give all panels the same y-axis range
y_max = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, y_max)
 

# ylabel only on left-side plots
axes[0].set_ylabel("Count (permutations)")
axes[3].set_ylabel("Count (permutations)")

sns.despine(top=True, right=True)
fig.suptitle(
    f"Permutation null distributions vs. observed AUC ({N_PERMUTATIONS} permutations, label-shuffled)",
    fontsize=11, y=0.98
)
plt.tight_layout()
plt.savefig(
    "plots/Prediction_Model/Logistic_Regression/5_Final/permutation_test.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("END OF SCRIPT")