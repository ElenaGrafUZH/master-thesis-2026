import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity

from scripts.plot_helper.colors import COLORS

import seaborn as sns
import matplotlib.pyplot as plt


#* ── Constants ────────────────────────────────────────────
OUTER_SPLITS = 5
OUTER_REPEATS = 50
PRIMARY_CUTOFF = 4.0
RANDOM_STATE = 42

SLEEP_PCA_COMPONENTS = 3
STEP_PCA_COMPONENTS = 3
HR_PCA_COMPONENTS = 2

out_dir = "output/4_Prediction_Model/3_Logistic_Regression/7_PCA_stability/"
plot_dir = "plots/Prediction_Model/Logistic_Regression/7_PCA_stability/"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

#* ── Functions ────────────────────────────────────────────
def make_pca_pipeline(n_components):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE))
    ])


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

num_features = len(df.columns)-1
num_participants = len(df)

print(f"Number of features: {num_features}, Number of participants: {num_participants}")


#* ── 1. Define Outcome and Feature Groups ────────────────────────────────────────────
target = "Matsuda_Index"


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

feature_sets = {
    "Sleep": {"features": sleep_features,
              "n_components": SLEEP_PCA_COMPONENTS},
    "Activity": {"features": step_features,
              "n_components": STEP_PCA_COMPONENTS},
    "HR/HRV": {"features": hr_features,
               "n_components": HR_PCA_COMPONENTS}
}

#* ── 2. Prepare Dataset ────────────────────────────────────────────
required_cols = (
    [target]
    + demo_features
    + sleep_features
    + step_features
    + hr_features
)

df_model = df[required_cols].dropna().copy()


X = df_model.drop(columns=[target])
y = (df_model[target] <= PRIMARY_CUTOFF).astype(int)

print(
    f"Participants used: {len(df_model)} "
    f"(dropped {len(df) - len(df_model)} with missing data)"
)
print(
    f"Primary target: class 1 = Matsuda Index <= {PRIMARY_CUTOFF}"
)
print("Class counts:")
print(y.value_counts().sort_index())

#* ── 3. PCA Stability Check across Outer-CV Folds ────────────────────────────────────────────
cv_diag = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)

pca_stability_results = []
fold_component_results = []
pairwise_similarity_results = []

for feature_set_name, settings in feature_sets.items():
    features = settings["features"]
    n_components = settings["n_components"]

    fold_loadings = []
    fold_evr = []  # explained variance ratio

    for iteration, (train_idx, _) in enumerate(cv_diag.split(X, y)):
        repeat = iteration // OUTER_SPLITS + 1
        fold = iteration % OUTER_SPLITS + 1

        X_train_pca = X.iloc[train_idx][features]
        fitted_pipeline = make_pca_pipeline(n_components).fit(X_train_pca)
        
        loadings = fitted_pipeline.named_steps['pca'].components_
        evr = fitted_pipeline.named_steps['pca'].explained_variance_ratio_

        fold_loadings.append(loadings)
        fold_evr.append(evr)

        for pc_idx in range(n_components):
            row = {
                "feature_set": feature_set_name,
                "component": f"PC{pc_idx + 1}",
                "repeat": repeat,
                "fold": fold,
                "fold_uid":iteration,
                "explained_variance_ratio": evr[pc_idx]
            }
            for feature_name, loading in zip(features, loadings[pc_idx]):
                row[f"loading__{feature_name}"] = loading
            fold_component_results.append(row)

    fold_loadings = np.array(fold_loadings)
    fold_evr = np.array(fold_evr)
    # compare PC1 loadings across folds via cosine similarity
    print("-" * 20)
    print(f"Feature set: {feature_set_name}")

    for pc_idx in range(n_components):
        pc_name = f"PC{pc_idx + 1}"

        pc_loadings = fold_loadings[:, pc_idx, :]

        # Absolute cosine similarity treats arbitrary sign reversals as
        # equivalent. Components are compared by component number, so low
        # similarity may partly reflect switching of similarly strong PCs
        # absolute cosine similarity accounts for arbitrary PCA sign flips
        sim_matrix = np.abs(cosine_similarity(pc_loadings))
        tri_i, tri_j = np.triu_indices_from(sim_matrix, k=1)
        pairwise_similarity = sim_matrix[tri_i, tri_j]

        for first_fit, second_fit, similarity in zip(
            tri_i,
            tri_j,
            pairwise_similarity,
        ):
            pairwise_similarity_results.append({
                "feature_set": feature_set_name,
                "component": pc_name,
                "fit_1": int(first_fit),
                "fit_2": int(second_fit),
                "absolute_cosine_similarity": similarity,
            })

        summary_row = {
            "feature_set": feature_set_name,
            "component": pc_name,
            "n_pca_fits": len(pc_loadings),
            "n_pairwise_comparisons": len(pairwise_similarity),
            "mean_cosine_similarity": pairwise_similarity.mean(),
            "median_cosine_similarity": np.median(pairwise_similarity),
            "sd_cosine_similarity": pairwise_similarity.std(ddof=0),
            "p05_cosine_similarity": np.percentile(pairwise_similarity, 5),
            "min_cosine_similarity": pairwise_similarity.min(),
            "mean_explained_variance_ratio": fold_evr[:, pc_idx].mean(),
            "sd_explained_variance_ratio": fold_evr[:, pc_idx].std(ddof=0),
        }
        pca_stability_results.append(summary_row)

        print(f"\n{pc_name}:")
        print(f"  Mean absolute cosine similarity: {summary_row['mean_cosine_similarity']:.3f}")
        print(f"  Median: {summary_row['median_cosine_similarity']:.3f}")
        print(f"  5th percentile: {summary_row['p05_cosine_similarity']:.3f}")
        print(f"  Minimum: {summary_row['min_cosine_similarity']:.3f}")
        print(
            "  Explained variance ratio: "
            f"{summary_row['mean_explained_variance_ratio']:.3f} "
            f"± {summary_row['sd_explained_variance_ratio']:.3f}"
        )



#* ── 4. Save Results ────────────────────────────────────────────
pca_stability_summary = pd.DataFrame(pca_stability_results)
fold_component_df = pd.DataFrame(fold_component_results)
pairwise_similarity_df = pd.DataFrame(pairwise_similarity_results)

pca_stability_summary.to_csv(
    out_dir + "pca_stability_summary_primary.csv",
    index=False,
)
fold_component_df.to_csv(
    out_dir + "pca_fold_loadings_and_evr_primary.csv",
    index=False,
)
pairwise_similarity_df.to_csv(
    out_dir + "pca_pairwise_cosine_similarity_primary.csv",
    index=False,
)

print("\nPCA stability summary:")
print(pca_stability_summary.to_string(index=False))

#* ── 5. Simple Stability Plot ────────────────────────────────────────────
# The dot shows the mean absolute cosine similarity. The vertical line spans
# the 5th percentile to the mean and is descriptive, not a confidence interval.
plot_df = pca_stability_summary.copy()
plot_df["component_number"] = (
    plot_df["component"].str.extract(r"(\d+)")[0].astype(int)
)

feature_order = ["Sleep", "Activity", "HR/HRV"]
feature_colors = {
    "Sleep": COLORS["blue1"],
    "Activity": COLORS["orange1"],
    "HR/HRV": COLORS["green1"],
}

fig, ax = plt.subplots(figsize=(8, 5))

x_positions = {
    "Sleep": np.array([1, 2, 3], dtype=float) - 0.12,
    "Activity": np.array([1, 2, 3], dtype=float),
    "HR/HRV": np.array([1, 2], dtype=float) + 0.12,
}

for feature_set in feature_order:
    sub = (
        plot_df[plot_df["feature_set"] == feature_set]
        .sort_values("component_number")
    )
    x = x_positions[feature_set][:len(sub)]

    ax.vlines(
        x=x,
        ymin=sub["p05_cosine_similarity"],
        ymax=sub["mean_cosine_similarity"],
        color=feature_colors[feature_set],
        linewidth=1.5,
        alpha=0.8,
    )
    ax.scatter(
        x,
        sub["mean_cosine_similarity"],
        color=feature_colors[feature_set],
        s=45,
        label=feature_set,
        zorder=3,
    )

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["PC1", "PC2", "PC3"])
ax.set_ylim(0, 1.03)
ax.set_xlabel("Principal component")
ax.set_ylabel("Absolute cosine similarity")
ax.set_title(
    "PCA loading stability across primary outer-CV training folds",
    fontsize=11,
)

ax.text(
    0.01,
    0.02,
    "Dots: mean; vertical ranges: 5th percentile to mean",
    transform=ax.transAxes,
    fontsize=8.5,
    color=COLORS["grey1"],
)

ax.legend(
    title="Feature domain",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
plt.savefig(
    plot_dir + "pca_loading_stability_primary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("END OF SCRIPT")