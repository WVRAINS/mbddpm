import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from xgboost import XGBClassifier

from mbddpm.utils.get_project_root import get_project_root


# ==================================================
# Reproducibility
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)



# ==================================================
# Load data
# ==================================================

def load_data(case_file, ctrl_file):

    case = pd.read_csv(case_file)
    ctrl = pd.read_csv(ctrl_file)

    if set(case.columns) != set(ctrl.columns):
        raise ValueError(
            "Feature columns between case and ctrl are inconsistent!"
        )

    ctrl = ctrl[case.columns]

    X = pd.concat(
        [case, ctrl],
        axis=0
    ).reset_index(drop=True)

    y = np.concatenate([
        np.ones(len(case)),
        np.zeros(len(ctrl))
    ])

    return X, y



# ==================================================
# Models
# ==================================================

def get_models():

    return {

        "LR":
        LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="liblinear",
            max_iter=2000,
            random_state=SEED
        ),


        "RF":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=SEED,
            n_jobs=-1
        ),


        "XGBoost":
        XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
            random_state=SEED
        )
    }



# ==================================================
# Evaluation
# ==================================================

def evaluate(model, X_train, y_train, X_test, y_test):

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    prob = model.predict_proba(
        X_test
    )[:, 1]


    return {

        "Accuracy":
        accuracy_score(
            y_test,
            pred
        ),

        "AUC":
        roc_auc_score(
            y_test,
            prob
        ),

        "F1":
        f1_score(
            y_test,
            pred
        ),

        "MCC":
        matthews_corrcoef(
            y_test,
            pred
        )
    }



# ==================================================
# Dataset
# ==================================================

real_X, real_y = load_data(
    "D:/common/document/GitHub/mbddpm/data/demo_IBD_case.csv",
    "D:/common/document/GitHub/mbddpm/data/demo_IBD_ctrl.csv"
)



sim_X, sim_y = load_data(
    "D:/common/document/GitHub/mbddpm/generated/dataset_IBD_case/epoch_150000_code_MB-DDPM_0805-202426.csv",
    "D:/common/document/GitHub/mbddpm/generated/dataset_IBD_ctrl/epoch_150000_code_MB-DDPM_0804-233713.csv"
)



# Feature consistency

if list(real_X.columns) != list(sim_X.columns):

    raise ValueError(
        "Real and synthetic feature columns are inconsistent!"
    )



# ==================================================
# 5-fold CV
# ==================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


results = []



for fold, (train_idx, test_idx) in enumerate(
        skf.split(real_X, real_y),
        start=1):


    print(
        f"Running fold {fold}"
    )


    # --------------------------
    # Real train/test split
    # --------------------------

    X_train = real_X.iloc[train_idx]
    y_train = real_y[train_idx]


    X_test = real_X.iloc[test_idx]
    y_test = real_y[test_idx]



    # --------------------------
    # Scaling
    # Fit only training data
    # --------------------------

    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    sim_X_scaled = scaler.transform(
        sim_X
    )



    # ==================================================
    # Experiment 1
    # Real only
    # ==================================================

    for name, model in get_models().items():

        metric = evaluate(
            model,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test
        )


        metric.update({

            "Fold": fold,

            "Model": name,

            "Experiment": "Real_only"

        })


        results.append(metric)



    # ==================================================
    # Experiment 2
    # Real + balanced synthetic
    # ==================================================

    n_case = np.sum(
        y_train == 1
    )

    n_ctrl = np.sum(
        y_train == 0
    )


    sim_case = sim_X_scaled[
        sim_y == 1
    ]


    sim_ctrl = sim_X_scaled[
        sim_y == 0
    ]



    rng = np.random.RandomState(
        SEED + fold
    )



    case_index = rng.choice(
        len(sim_case),
        size=n_case,
        replace=False
    )


    ctrl_index = rng.choice(
        len(sim_ctrl),
        size=n_ctrl,
        replace=False
    )



    sim_bal_X = np.vstack([

        sim_case[case_index],

        sim_ctrl[ctrl_index]

    ])


    sim_bal_y = np.concatenate([

        np.ones(n_case),

        np.zeros(n_ctrl)

    ])



    X_aug = np.vstack([

        X_train_scaled,

        sim_bal_X

    ])



    y_aug = np.concatenate([

        y_train,

        sim_bal_y

    ])



    X_aug, y_aug = shuffle(
        X_aug,
        y_aug,
        random_state=SEED
    )



    for name, model in get_models().items():

        metric = evaluate(
            model,
            X_aug,
            y_aug,
            X_test_scaled,
            y_test
        )


        metric.update({

            "Fold": fold,

            "Model": name,

            "Experiment":
            "Real_plus_Synthetic_Balanced"

        })


        results.append(metric)



    # ==================================================
    # Experiment 3
    # Synthetic only (TSTR)
    # ==================================================

    for name, model in get_models().items():

        metric = evaluate(
            model,
            sim_X_scaled,
            sim_y,
            X_test_scaled,
            y_test
        )


        metric.update({

            "Fold": fold,

            "Model": name,

            "Experiment":
            "Synthetic_only_TSTR"

        })


        results.append(metric)



# ==================================================
# Save results
# ==================================================

result_df = pd.DataFrame(results)


result_dir = get_project_root() / "experiment"

result_dir.mkdir(
    exist_ok=True
)



result_df.to_csv(
    result_dir / "classification_5fold_results.csv",
    index=False
)



summary = (

    result_df
    .groupby(
        ["Model", "Experiment"]
    )
    [
        [
            "Accuracy",
            "AUC",
            "F1",
            "MCC"
        ]
    ]
    .agg(
        ["mean", "std"]
    )

)



summary.to_csv(
    result_dir / "classification_5fold_summary.csv"
)



# ==================================================
# Plot 1
# AUC barplot
# ==================================================

sns.set_theme(
    style="whitegrid",
    font_scale=1.2
)


plt.figure(
    figsize=(12,6)
)



ax = sns.barplot(
    data=result_df,
    x="Model",
    y="AUC",
    hue="Experiment",
    errorbar="sd",
    palette="Set2",
    capsize=0.08
)


ax.set_ylabel(
    "AUC (5-fold CV)",
    fontsize=14
)


ax.set_xlabel(
    ""
)


ax.set_title(
    "Downstream classification performance",
    fontsize=16
)


plt.xticks(
    fontsize=12
)


plt.legend(
    title="Experiment",
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False
)


sns.despine()


plt.tight_layout()


plt.savefig(
    result_dir / "classification_AUC_comparison.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# ==================================================
# Plot 2
# AUC heatmap
# ==================================================

auc_mean = (

    result_df
    .groupby(
        ["Model","Experiment"]
    )["AUC"]
    .mean()
    .reset_index()

)



auc_matrix = auc_mean.pivot(
    index="Model",
    columns="Experiment",
    values="AUC"
)



auc_matrix.columns = [

    "Real",

    "Real+Balanced",

    "Synthetic(TSTR)"

]



plt.figure(
    figsize=(8,5)
)



sns.heatmap(
    auc_matrix,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={
        "label":"Mean AUC"
    }
)



plt.title(
    "Mean AUC across 5-fold CV",
    fontsize=15
)


plt.xlabel("")

plt.ylabel("")


plt.xticks(
    rotation=45,
    ha="right"
)



plt.tight_layout()



plt.savefig(
    result_dir / "classification_AUC_heatmap.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



print("\n===== Summary =====")
print(summary)


print(
    f"\nResults saved in: {result_dir}"
)