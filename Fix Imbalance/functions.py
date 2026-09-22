import io
import subprocess
from collections import Counter, defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Newer XGBoost stores base_score bracket-wrapped ("[0.294]"); shap calls
# float() on it directly and crashes, so patch shap's decoder to strip it.
_orig_decode_ubjson = shap.explainers._tree.decode_ubjson_buffer


def _patched_decode_ubjson(fd):
    result = _orig_decode_ubjson(fd)
    try:
        bs = result["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
            result["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
    except (KeyError, TypeError):
        pass
    return result


shap.explainers._tree.decode_ubjson_buffer = _patched_decode_ubjson
import os

from matplotlib import font_manager


def is_latex_available() -> bool:
    """Whether a working LaTeX + cm-super/type1cm toolchain is installed for matplotlib's usetex."""
    try:
        subprocess.run(["latex", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    for sty in ("type1cm.sty", "type1ec.sty"):
        try:
            r = subprocess.run(["kpsewhich", sty], capture_output=True, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        if r.returncode != 0 or not r.stdout.strip():
            return False
    return True


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/final", exist_ok=True)
    os.makedirs("./results/final/roc_curves", exist_ok=True)


plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.labelsize"] = (
    19  # larger axis labels (normal weight) for readability
)
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["xtick.labelsize"] = 18  # larger tick labels for readability
plt.rcParams["ytick.labelsize"] = 18

# Publication figure style: Type-42 embedded fonts, sans-serif, no LaTeX (journal requirement).
plt.rcParams.update(
    {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        # Arial first, then Arial-metric fallbacks (see requirements.txt for the install command).
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Helvetica",
            "Nimbus Sans",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "dejavusans",  # sans-serif math symbols (e.g. $\pm$)
        "mathtext.default": "regular",
    }
)

# width as measured in inkscape
width = 3.45
height = width / 1.618
plt.rcParams["figure.figsize"] = (width, height)


# Pinned waterfall selections used in the manuscript figures.
PINNED_WATERFALLS = {
    ("knn_age_matched_mm - AD vs Healthy", 0, 4): {
        "Correctly Classified as AD": [169],  # Fig9a  (f(x)=-1.693)
        "Correctly Classified as CN": [92],  # Fig9b  (f(x)=-4.005)
        "Misclassified AD as CN": [479],  # Fig10a (f(x)=-3.335)
        "Misclassified CN as AD": [371],  # Fig10b (f(x)=-1.386)
    },
}


def macro_sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the macro sensitivity and specificity for a multiclass classification task.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Tuple containing macro sensitivity and specificity
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    total = cm.sum()
    sensitivities = []
    specificities = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - (TP + FN + FP)
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)
    macro_sens = sum(sensitivities) / num_classes
    macro_spec = sum(specificities) / num_classes
    return macro_sens, macro_spec


def sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the sensitivity and specificity for a binary classification task.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Tuple containing sensitivity and specificity
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    return sensitivity, specificity


def get_ad_years(year: int) -> pd.Series:
    """
    Get the eids of patients diagnosed with Alzheimer's Disease within a certain number of years.

    Args:
        year: Number of years to consider

    Returns:
        Series containing eids of patients
    """
    ad_years = pd.read_csv(r"ad_years.csv")
    ad_years = ad_years[ad_years["ad_after0"] <= year]
    return ad_years["eid"]


def make_prediction(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    name: str = "",
) -> None:
    """
    Train and evaluate two XGBoost classifiers (with and without class weights).

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        name: Name for saving results
    """
    ensure_directories()

    # Calculate instance weights based on class distribution
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # Train XGBoost classifiers
    model_1 = XGBClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        enable_categorical=True,
    )
    model_2 = XGBClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        enable_categorical=True,
    )
    model_1.fit(x_train, y_train, sample_weight=sample_weight)
    model_2.fit(x_train, y_train)

    # Prediction
    y_pred_weight = model_1.predict(x_test)
    y_pred = model_2.predict(x_test)

    # Prepare a buffer to capture print statements
    with io.StringIO() as buffer:
        # Model evaluation - Train Set
        print(f"Model Performance on Train Set - {name}", file=buffer)
        print(
            "ROC AUC Score for weighted model on train set:",
            roc_auc_score(y_train, model_1.predict_proba(x_train), multi_class="ovr"),
            file=buffer,
        )
        print(
            "ROC AUC Score for unweighted model on train set:",
            roc_auc_score(y_train, model_2.predict_proba(x_train), multi_class="ovr"),
            file=buffer,
        )
        print(
            "Matthews Correlation Coefficient for weighted model on train set:",
            matthews_corrcoef(y_train, model_1.predict(x_train)),
            file=buffer,
        )
        print(
            "Matthews Correlation Coefficient for unweighted model on train set:",
            matthews_corrcoef(y_train, model_2.predict(x_train)),
            file=buffer,
        )
        print(
            "Macro F1 Score for weighted model on train set:",
            f1_score(
                y_train, model_1.predict(x_train), average="macro", zero_division=1
            ),
            file=buffer,
        )
        print(
            "Macro F1 Score for unweighted model on train set:",
            f1_score(
                y_train, model_2.predict(x_train), average="macro", zero_division=1
            ),
            file=buffer,
        )
        print(
            "Accuracy Score for weighted model on train set:",
            accuracy_score(y_train, model_1.predict(x_train)),
            file=buffer,
        )
        print(
            "Accuracy Score for unweighted model on train set:",
            accuracy_score(y_train, model_2.predict(x_train)),
            file=buffer,
        )
        # Compute and print macro sensitivity and specificity for the train set
        macro_sens_train_weighted, macro_spec_train_weighted = (
            macro_sensitivity_specificity(y_train, model_1.predict(x_train))
        )
        print(
            "Macro Sensitivity for weighted model on train set:",
            macro_sens_train_weighted,
            file=buffer,
        )
        print(
            "Macro Specificity for weighted model on train set:",
            macro_spec_train_weighted,
            file=buffer,
        )

        macro_sens_train_unweighted, macro_spec_train_unweighted = (
            macro_sensitivity_specificity(y_train, model_2.predict(x_train))
        )
        print(
            "Macro Sensitivity for unweighted model on train set:",
            macro_sens_train_unweighted,
            file=buffer,
        )
        print(
            "Macro Specificity for unweighted model on train set:",
            macro_spec_train_unweighted,
            file=buffer,
        )

        print("-" * 50, file=buffer)

        # Model evaluation - Test Set
        print(f"Model Performance on Test Set - {name}", file=buffer)
        print(
            "ROC AUC Score for weighted model on test set:",
            roc_auc_score(y_test, model_1.predict_proba(x_test), multi_class="ovr"),
            file=buffer,
        )
        print(
            "ROC AUC Score for unweighted model on test set:",
            roc_auc_score(y_test, model_2.predict_proba(x_test), multi_class="ovr"),
            file=buffer,
        )
        print(
            "Matthews Correlation Coefficient for weighted model:",
            matthews_corrcoef(y_test, y_pred_weight),
            file=buffer,
        )
        print(
            "Matthews Correlation Coefficient for unweighted model:",
            matthews_corrcoef(y_test, y_pred),
            file=buffer,
        )
        print(
            "Macro F1 Score for weighted model on test set:",
            f1_score(y_test, y_pred_weight, average="macro", zero_division=1),
            file=buffer,
        )
        print(
            "Macro F1 Score for unweighted model on test set:",
            f1_score(y_test, y_pred, average="macro", zero_division=1),
            file=buffer,
        )
        print(
            "Accuracy Score for weighted model on test set:",
            accuracy_score(y_test, y_pred_weight),
            file=buffer,
        )
        print(
            "Accuracy Score for unweighted model on test set:",
            accuracy_score(y_test, y_pred),
            file=buffer,
        )

        # Compute and print macro sensitivity and specificity for the test set
        macro_sens_test_weighted, macro_spec_test_weighted = (
            macro_sensitivity_specificity(y_test, y_pred_weight)
        )
        macro_sens_test_unweighted, macro_spec_test_unweighted = (
            macro_sensitivity_specificity(y_test, y_pred)
        )

        print(
            "Macro Sensitivity for weighted model on test set:",
            macro_sens_test_weighted,
            file=buffer,
        )
        print(
            "Macro Sensitivity for unweighted model on test set:",
            macro_sens_test_unweighted,
            file=buffer,
        )

        print(
            "Macro Specificity for weighted model on test set:",
            macro_spec_test_weighted,
            file=buffer,
        )
        print(
            "Macro Specificity for unweighted model on test set:",
            macro_spec_test_unweighted,
            file=buffer,
        )

        # Classification Reports
        print("\nXGBoost Classification Report for Weighted Model:", file=buffer)
        print(
            classification_report(
                y_test, y_pred_weight, target_names=["healthy", "dementia", "AD"]
            ),
            file=buffer,
        )
        print("\nXGBoost Classification Report for Unweighted Model:", file=buffer)
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["healthy", "dementia", "AD"],
                zero_division=1,
            ),
            file=buffer,
        )

        # Save results to PDF
        with PdfPages(f"./results/{name}.pdf", "w") as pdf:
            # Page 1: Text Metrics
            buffer.seek(0)
            results_text = buffer.getvalue()
            fig = plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.text(
                0,
                1,
                results_text,
                verticalalignment="top",
                fontsize=10,
                fontfamily="monospace",
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Confusion Matrices
            conf_matrix_weighted = confusion_matrix(y_test, y_pred_weight)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(
                conf_matrix_weighted,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=model_1.classes_,
                yticklabels=model_1.classes_,
                ax=axes[0],
            )
            axes[0].set_title("Confusion Matrix (Weighted)")
            axes[0].set_xlabel("Predicted Class")
            axes[0].set_ylabel("True Class")

            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=model_2.classes_,
                yticklabels=model_2.classes_,
                ax=axes[1],
            )
            axes[1].set_title("Confusion Matrix (Unweighted)")
            axes[1].set_xlabel("Predicted Class")
            axes[1].set_ylabel("True Class")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Feature Importances
            with io.StringIO() as feature_buffer:
                print("Feature Importances", file=feature_buffer)
                feature_importances = pd.DataFrame(
                    {
                        "Feature": x_train.columns,
                        "Importance": model_2.feature_importances_,
                    }
                ).sort_values(by="Importance", ascending=False)
                pd.set_option("display.max_colwidth", None)
                feature_importances = feature_importances.round(5).to_string(
                    index=False
                )
                print(feature_importances, file=feature_buffer)
                feature_importances_text = feature_buffer.getvalue()

                fig = plt.figure(figsize=(10, 10))
                plt.axis("off")
                plt.text(
                    0,
                    1,
                    feature_importances_text,
                    verticalalignment="top",
                    fontsize=10,
                    fontfamily="monospace",
                )
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def make_prediction_2class(
    data: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    downsample: bool = False,
    upsample: bool = False,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    name: str = "",
    isfinal: bool = False,
    target: str = "dementia",
    isShapley: bool = False,
    output_path: str = "./results/",
    is_csv: bool = False,
    random_state=41,
) -> None:
    """
    Train and evaluate two XGBoost classifiers for binary classification.

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        name: Name for saving results
        isfinal: Whether this is the final model
        target: Target class for the binary classification
    """
    ensure_directories()

    metrics = {
        "roc_auc_train_weighted": [],
        "roc_auc_train_unweighted": [],
        "mcc_train_weighted": [],
        "mcc_train_unweighted": [],
        "f1_train_weighted": [],
        "f1_train_unweighted": [],
        "accuracy_train_weighted": [],
        "accuracy_train_unweighted": [],
        "auprc_train_weighted": [],
        "auprc_train_unweighted": [],
        "sensitivity_train_weighted": [],
        "sensitivity_train_unweighted": [],
        "specificity_train_weighted": [],
        "specificity_train_unweighted": [],
    }

    y_preds = []
    y_probas = []
    y_pred_weights = []
    y_probas_weights = []
    feature_importance_list = []
    shap_values_org = []
    df = data.copy()
    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    test_eids = []

    # Initialize StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # Prepare X and y from the full dataset
    X = df.drop(columns=["Status"])
    y = df["Status"]

    # get class counts
    count_class0, count_class1 = df["Status"].value_counts()

    # Iterate through each fold
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        # Split the data using the fold indices
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        num_cols = [col for col in num_features if col in X_train.columns]
        train_min = X_train[num_cols].min()
        train_max = X_train[num_cols].max()

        X_train[num_cols] = (X_train[num_cols] - train_min) / (
            train_max - train_min + 1e-9
        )
        X_test[num_cols] = (X_test[num_cols] - train_min) / (
            train_max - train_min + 1e-9
        )

        if downsample:
            # custom cluster undersampling
            if len(y_train) >= 5000:
                # custom cluster undersampling
                X_train, y_train = custom_cluster_undersampling(
                    X_train, y_train, cat_features, num_features, 5000
                )

        if upsample:
            # then reduce the borderline majority class using TomekLinks
            tomek = TomekLinks()
            X_train, y_train = tomek.fit_resample(X_train, y_train)
            cat_indices = [
                X_train.columns.get_loc(col)
                for col in cat_features
                if col in X_train.columns
            ]
            # upsampling with smote-nc 5k data, 5x sample, tomeklinks
            smote_nc = SMOTENC(
                categorical_features=cat_indices,
                random_state=random_state + fold_idx,
                sampling_strategy={0: 5000, 1: count_class1 * 5},
            )
            X_train, y_train = smote_nc.fit_resample(X_train, y_train)
            print(
                "Resampled class distribution after SMOTENC and tomeklinks:",
                Counter(y_train),
            )

        # Calculate instance weights based on class distribution
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        # Store the splits
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)
        test_eids.append(X_test.index.tolist())

        # Train models
        model_1 = XGBClassifier(
            random_state=random_state + fold_idx,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            enable_categorical=True,
        )

        model_2 = XGBClassifier(
            random_state=random_state + fold_idx,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            enable_categorical=True,
        )

        model_1.fit(X_train, y_train, sample_weight=sample_weight)
        model_2.fit(X_train, y_train)

        y_pred_weight = model_1.predict(X_test)
        y_pred = model_2.predict(X_test)
        y_proba_weight = model_1.predict_proba(X_test)[:, 1]
        y_proba = model_2.predict_proba(X_test)[:, 1]

        model_1_train_predict = model_1.predict(X_train)
        model_2_train_predict = model_2.predict(X_train)
        y_train_proba_weight = model_1.predict_proba(X_train)[:, 1]
        y_train_proba = model_2.predict_proba(X_train)[:, 1]

        y_preds.append(y_pred)
        y_probas.append(y_proba)
        y_pred_weights.append(y_pred_weight)
        y_probas_weights.append(y_proba_weight)

        # Collect metrics

        metrics["mcc_train_weighted"].append(
            matthews_corrcoef(y_train, model_1_train_predict)
        )
        metrics["mcc_train_unweighted"].append(
            matthews_corrcoef(y_train, model_2_train_predict)
        )
        metrics["f1_train_weighted"].append(
            f1_score(y_train, model_1_train_predict, average="macro", zero_division=1)
        )
        metrics["f1_train_unweighted"].append(
            f1_score(y_train, model_2_train_predict, average="macro", zero_division=1)
        )
        metrics["roc_auc_train_weighted"].append(
            roc_auc_score(y_train, y_train_proba_weight)
        )
        metrics["roc_auc_train_unweighted"].append(
            roc_auc_score(y_train, y_train_proba)
        )
        metrics["accuracy_train_weighted"].append(
            accuracy_score(y_train, model_1_train_predict)
        )
        metrics["accuracy_train_unweighted"].append(
            accuracy_score(y_train, model_2_train_predict)
        )
        metrics["auprc_train_weighted"].append(
            average_precision_score(y_train, y_train_proba_weight)
        )
        metrics["auprc_train_unweighted"].append(
            average_precision_score(y_train, y_train_proba)
        )

        sens_train_w, spec_train_w = sensitivity_specificity(
            y_train, model_1_train_predict
        )
        sens_train_u, spec_train_u = sensitivity_specificity(
            y_train, model_2_train_predict
        )
        metrics["sensitivity_train_weighted"].append(sens_train_w)
        metrics["sensitivity_train_unweighted"].append(sens_train_u)
        metrics["specificity_train_weighted"].append(spec_train_w)
        metrics["specificity_train_unweighted"].append(spec_train_u)

        # Collect feature importances
        feature_importance_list.append(model_2.feature_importances_)

        # Collect SHAP values if required
        if isShapley:
            explainer = shap.TreeExplainer(model_2)
            shap_values = explainer(X_test)
            shap_values_org.append(shap_values)

        # Save each fold's train/test data if is_csv is True
        if is_csv:
            if upsample:
                # save data for upsampled configuration
                path = output_path + f"final_datasets/{name}_5k_5x_ad/fold_{fold_idx}/"
                os.makedirs(path, exist_ok=True)
                df_train_resampled = pd.concat([X_train, y_train], axis=1)
                df_test_fold = pd.concat([X_test, y_test], axis=1)
                df_train_resampled.to_csv(path + "train.csv", index=False)
                df_test_fold.to_csv(path + "test.csv", index=False)

            elif downsample:
                # save data for downsampled configuration
                path = output_path + f"final_datasets/{name}_5k_ad/fold_{fold_idx}/"
                os.makedirs(path, exist_ok=True)
                df_train_resampled = pd.concat([X_train, y_train], axis=1)
                df_test_fold = pd.concat([X_test, y_test], axis=1)
                df_train_resampled.to_csv(path + "train.csv", index=False)
                df_test_fold.to_csv(path + "test.csv", index=False)

            else:
                # save data for standard configuration
                path = output_path + f"final_datasets/{name}_{target}/fold_{fold_idx}/"
                os.makedirs(path, exist_ok=True)
                df_train_final = pd.concat([X_train, y_train], axis=1)
                df_test_final = pd.concat([X_test, y_test], axis=1)
                df_train_final.to_csv(path + "train.csv", index=False)
                df_test_final.to_csv(path + "test.csv", index=False)

    # Keep as plain lists so np.concatenate produces proper numeric dtypes
    # (wrapping with dtype=object causes sklearn to reject them as "unknown" targets)

    feature_importance_list = np.array(feature_importance_list)

    return {
        "metrics": metrics,
        "y_preds": y_preds,
        "y_probas": y_probas,
        "y_pred_weights": y_pred_weights,
        "y_probas_weights": y_probas_weights,
        "y_tests": y_tests,
        "feature_importances": feature_importance_list,
        "shap_values": shap_values_org if isShapley else None,
        "feature_names": X_train.columns.tolist(),
        "test_eids": test_eids,
    }


def custom_cluster_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    object_columns: List[str],
    numeric_columns: List[str],
    n_clusters: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform custom cluster-based undersampling.

    Args:
        X_train: Training features
        y_train: Training labels
        object_columns: List of categorical column names
        numeric_columns: List of numeric column names
        n_clusters: Number of clusters to create

    Returns:
        Tuple containing resampled features and labels
    """
    # separate the majority and minority classes
    X_majority = X_train[y_train == 0]
    X_minority = X_train[y_train != 0]
    y_minority = y_train[y_train != 0]

    X_majority_numerical = X_majority[numeric_columns]

    # cluster centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    X_numerical_centroids = kmeans.fit(X_majority_numerical).cluster_centers_

    X_majority = X_majority.copy()
    # majority voting for object columns
    X_majority.loc[:, "cluster"] = kmeans.labels_
    X_categorical_resampled = X_majority.groupby("cluster")[object_columns].agg(
        lambda x: x.mode()[0]
    )  # majority voting

    # combine numerical and object columns
    X_resampled = pd.concat(
        [
            pd.DataFrame(X_numerical_centroids, columns=numeric_columns),
            X_categorical_resampled.reset_index(drop=True),
        ],
        axis=1,
    )

    # assign Labels
    y_resampled_majority = np.zeros(
        X_resampled.shape[0], dtype=int
    )  # assign majority class label (0)
    X_resampled = pd.concat(
        [X_resampled, X_minority]
    )  # add minority class samples back
    y_resampled = np.concatenate([y_resampled_majority, y_minority])  # combine labels

    # reorder the columns
    X_resampled = X_resampled[X_train.columns]

    # transform y_resampled into a Pandas Series
    y_resampled_series = pd.Series(y_resampled, index=X_resampled.index, name="Status")

    return X_resampled, y_resampled_series


def mask_feature_value(label_text):
    parts = label_text.split(" = ", 1)
    if len(parts) != 2:
        return label_text
    value_str, feature_name = parts[0].strip(), parts[1].strip()
    try:
        float(value_str)
    except ValueError:
        return label_text
    masked_value = ""
    if "." in value_str:
        integer_part, decimal_part = value_str.split(".")
        if decimal_part:
            kept_part = f"{integer_part}.{decimal_part[0]}"
            num_masked_chars = len(decimal_part) - 1
            masked_value = kept_part + "*" * num_masked_chars
        else:
            masked_value = f"{integer_part}."
    else:
        if len(value_str) == 1:
            masked_value = "*"
        else:
            masked_value = value_str[0] + "*" * (len(value_str) - 1)
    return f"{masked_value} = {feature_name}"


def visualize_combined_results(
    metrics: dict,
    y_preds: list,
    y_probas: list,
    y_pred_weights: list,
    y_probas_weights: list,
    y_tests: list,
    feature_importances: list,
    shap_values: list = None,
    feature_names: list = None,
    output_path: str = "./results/",
    name: str = "combined_result",
    test_eids: list = None,
    target_names: list = ["Normal", "Dementia"],
    isfinal: bool = True,
    shap_values_per_fold: list = None,
    y_tests_per_fold: list = None,
    y_probas_per_fold: list = None,
):
    os.makedirs(output_path, exist_ok=True)
    pdf_path = os.path.join(output_path, f"{'final/' if isfinal else ''}{name}.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        with io.StringIO() as buffer:
            print(f"Aggregated Results for {name}\n", file=buffer)
            for key, values in metrics.items():
                mean = np.mean(values)
                std = np.std(values)
                print(f"{key}: {mean:.4f} +/- {std:.4f}", file=buffer)
                if key == "specificity_train_unweighted":
                    print("\n", file=buffer)
                    print("-" * 50, file=buffer)
                    print("\n", file=buffer)

            buffer.seek(0)
            _prev_usetex = plt.rcParams.get("text.usetex", False)
            plt.rcParams["text.usetex"] = False
            fig = plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.text(
                0,
                1,
                buffer.read(),
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=9,
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            plt.rcParams["text.usetex"] = _prev_usetex

            # Confusion Matrices — average across datasets
            cms = [confusion_matrix(yt, yp) for yt, yp in zip(y_tests, y_preds)]
            cms_w = [
                confusion_matrix(yt, yp) for yt, yp in zip(y_tests, y_pred_weights)
            ]
            cm = np.mean(cms, axis=0).round().astype(int)
            cm_w = np.mean(cms_w, axis=0).round().astype(int)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            sns.heatmap(
                cm_w,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[0],
                xticklabels=target_names,
                yticklabels=target_names,
            )
            axes[0].set_title("Confusion Matrix (Weighted)")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True")

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Oranges",
                ax=axes[1],
                xticklabels=target_names,
                yticklabels=target_names,
            )
            axes[1].set_title("Confusion Matrix (Unweighted)")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")

            plt.tight_layout()

            if isfinal:
                cnf_path = (
                    output_path + f"final/confusion_matrices/{name}_confusion.pdf"
                )
                os.makedirs(os.path.dirname(cnf_path), exist_ok=True)
                fig.savefig(cnf_path, bbox_inches="tight")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            mean_fpr = np.linspace(0, 1, 500)
            target_fpr = 0.15
            tpr_weighted_list, tpr_unweighted_list = [], []
            auc_weighted_list, auc_unweighted_list = [], []

            for y_true, y_prob_w, y_prob in zip(y_tests, y_probas_weights, y_probas):
                y_true = np.asarray(y_true).astype(np.int32).ravel()
                y_prob_w = np.asarray(y_prob_w).astype(np.float64).ravel()
                y_prob = np.asarray(y_prob).astype(np.float64).ravel()

                fpr_w, tpr_w, _ = roc_curve(y_true, y_prob_w)
                fpr_u, tpr_u, _ = roc_curve(y_true, y_prob)

                fpr_w = np.asarray(fpr_w).astype(np.float64).ravel()
                tpr_w = np.asarray(tpr_w).astype(np.float64).ravel()
                fpr_u = np.asarray(fpr_u).astype(np.float64).ravel()
                tpr_u = np.asarray(tpr_u).astype(np.float64).ravel()

                tpr_weighted_interp = np.interp(mean_fpr, fpr_w, tpr_w)
                tpr_unweighted_interp = np.interp(mean_fpr, fpr_u, tpr_u)

                tpr_weighted_list.append(tpr_weighted_interp)
                tpr_unweighted_list.append(tpr_unweighted_interp)

                auc_weighted_list.append(auc(fpr_w, tpr_w))
                auc_unweighted_list.append(auc(fpr_u, tpr_u))

            mean_tpr_w = np.mean(tpr_weighted_list, axis=0)
            mean_tpr_u = np.mean(tpr_unweighted_list, axis=0)
            std_tpr_w = np.std(tpr_weighted_list, axis=0)
            std_tpr_u = np.std(tpr_unweighted_list, axis=0)
            mean_auc_w = np.mean(auc_weighted_list).round(3)
            mean_auc_u = np.mean(auc_unweighted_list).round(3)
            std_auc_w = np.std(auc_weighted_list).round(3)
            std_auc_u = np.std(auc_unweighted_list).round(3)

            tpr_weighted_at_target = np.interp(target_fpr, mean_fpr, mean_tpr_w).round(
                3
            )
            std_weighted_at_target = np.interp(target_fpr, mean_fpr, std_tpr_w).round(3)
            tpr_unweighted_at_target = np.interp(
                target_fpr, mean_fpr, mean_tpr_u
            ).round(3)
            std_unweighted_at_target = np.interp(target_fpr, mean_fpr, std_tpr_u).round(
                3
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                mean_fpr,
                mean_tpr_w,
                label=f"W Mean ROC (mAUC = {mean_auc_w} $\\pm$ {std_auc_w})",
                color="blue",
            )
            ax.fill_between(
                mean_fpr,
                mean_tpr_w - std_tpr_w,
                mean_tpr_w + std_tpr_w,
                alpha=0.2,
                color="blue",
            )

            ax.plot(
                mean_fpr,
                mean_tpr_u,
                label=f"U Mean ROC (mAUC = {mean_auc_u} $\\pm$ {std_auc_u})",
                color="orange",
            )
            ax.fill_between(
                mean_fpr,
                mean_tpr_u - std_tpr_u,
                mean_tpr_u + std_tpr_u,
                alpha=0.2,
                color="orange",
            )

            ax.axvline(
                x=target_fpr,
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="FPR = 0.15",
            )
            ax.scatter(
                target_fpr,
                tpr_weighted_at_target,
                color="blue",
                s=150,
                marker="*",
                zorder=5,
                label=f"W mTPR={tpr_weighted_at_target} $\\pm$ {std_weighted_at_target}",
            )
            ax.scatter(
                target_fpr,
                tpr_unweighted_at_target,
                color="orange",
                s=150,
                marker="*",
                zorder=5,
                label=f"U mTPR={tpr_unweighted_at_target} $\\pm$ {std_unweighted_at_target}",
            )

            ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
            ax.legend(loc="lower right")
            ax.grid(True)

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Mean ROC Curve")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if isfinal:
                # # CSV
                roc_df = pd.DataFrame(
                    {
                        "False Positive Rate": mean_fpr,
                        "Mean True Positive Rate": mean_tpr_u,
                        "Std True Positive Rate": std_tpr_u,
                        "ROC Score": mean_auc_u,
                        "Std roc score": std_auc_u,
                        "Mean tpr at target": tpr_unweighted_at_target,
                        "Std tpr at target": std_unweighted_at_target,
                    }
                )
                os.makedirs(output_path + "final/roc_curves/", exist_ok=True)
                roc_df.to_csv(
                    output_path + f"final/roc_curves/{name}_roc_curve.csv", index=False
                )

            # 4. Feature Importance

            os.makedirs(
                output_path + f"{'final/' if isfinal else ''}feature_importances/",
                exist_ok=True,
            )

            mean_importance = np.mean(feature_importances, axis=0)

            with io.StringIO() as feature_buffer:
                print("Feature Importances", file=feature_buffer)
                feature_importances_df = pd.DataFrame(
                    {"Feature": feature_names, "Mean Importance": mean_importance}
                ).sort_values(by="Mean Importance", ascending=False)

                pd.set_option("display.max_colwidth", None)

                feature_importances = feature_importances_df.round(5).copy()
                feature_importances_string = feature_importances.to_string(index=False)
                print(feature_importances_string, file=feature_buffer)
                feature_importances_text = feature_buffer.getvalue()

                _prev_usetex = plt.rcParams.get("text.usetex", False)
                plt.rcParams["text.usetex"] = False
                fig = plt.figure(figsize=(10, 10))
                plt.axis("off")
                plt.text(
                    0,
                    1,
                    feature_importances_text,
                    verticalalignment="top",
                    fontsize=10,
                    fontfamily="monospace",
                )
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                plt.rcParams["text.usetex"] = _prev_usetex

                top_n = 15
                if len(feature_importances_df) < top_n:
                    feature_importances_df_top = feature_importances_df.copy()

                else:
                    feature_importances_df_top = feature_importances_df.head(
                        top_n
                    ).copy()
                    feature_importances_df_top.to_csv(
                        output_path
                        + f"{'final/' if isfinal else ''}feature_importances/{name}_top_features.csv",
                        index=False,
                    )
                    feature_importances.to_csv(
                        output_path
                        + f"{'final/' if isfinal else ''}feature_importances/{name}_all_features.csv",
                        index=False,
                    )

            if shap_values is not None:
                combined_shap_values = np.vstack([exp.values for exp in shap_values])
                combined_data = np.vstack([exp.data for exp in shap_values])
                combined_base_values = np.concatenate(
                    [exp.base_values for exp in shap_values]
                )
                feature_names_shap = shap_values[0].feature_names

                # Average SHAP values per unique participant to avoid pseudo-replication
                if test_eids is not None:
                    all_eids_flat = sum(test_eids, [])
                    eid_arr = np.array(all_eids_flat)
                    unique_eids = np.unique(eid_arr)
                    avg_shap = np.zeros(
                        (len(unique_eids), combined_shap_values.shape[1])
                    )
                    avg_data = np.zeros((len(unique_eids), combined_data.shape[1]))
                    avg_base = np.zeros(len(unique_eids))
                    for i, eid in enumerate(unique_eids):
                        mask = eid_arr == eid
                        avg_shap[i] = combined_shap_values[mask].mean(axis=0)
                        avg_data[i] = combined_data[mask].mean(axis=0)
                        avg_base[i] = combined_base_values[mask].mean()
                    combined_shap_values = avg_shap
                    combined_data = avg_data
                    combined_base_values = avg_base

                global_mean_shap = np.abs(combined_shap_values).mean(axis=0)

                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_names_shap,
                        "Mean Absolute SHAP Value": global_mean_shap,
                    }
                ).sort_values(by="Mean Absolute SHAP Value", ascending=False)

                top_idx = np.argsort(global_mean_shap)[::-1][:15]

                shap_top = shap.Explanation(
                    values=combined_shap_values[:, top_idx],
                    base_values=combined_base_values,
                    data=combined_data[:, top_idx],
                    feature_names=np.array(feature_names_shap)[top_idx],
                )

                shap_full = shap.Explanation(
                    values=combined_shap_values,
                    base_values=combined_base_values,
                    data=combined_data,
                    feature_names=feature_names_shap,
                )

                shap_1 = shap.Explanation(
                    values=shap_values[1].values,
                    base_values=shap_values[1].base_values,
                    data=shap_values[1].data,
                    feature_names=shap_values[1].feature_names,
                )

                # SHAP draws its own text (feature labels, ± values with a Unicode minus) that
                # bare LaTeX can't typeset; force usetex off for the SHAP beeswarm plots, restore after.
                _prev_usetex_shap = plt.rcParams.get("text.usetex", False)
                plt.rcParams["text.usetex"] = False

                # create figure first
                plt.figure(figsize=(25, 30))

                # draw shap beeswarm plot
                shap.plots.beeswarm(shap_top, show=False, max_display=15)
                plt.gcf().set_size_inches(40, 20)
                # grab the current axis
                ax = plt.gca()

                tick_font = font_manager.FontProperties(weight="normal", size=35)

                # set labels and tick params
                ax.set_xlabel("Mean SHAP Value", fontsize=40, fontweight="normal")
                ax.set_ylabel("Feature", fontsize=40, fontweight="normal")
                ax.tick_params(axis="both", which="major", labelsize=35)
                ax.set_xticks(np.arange(-1, 1.1, 0.5))
                ax.set_xlim(-1, 1)
                plt.setp(ax.get_xticklabels(), fontproperties=tick_font)
                plt.setp(ax.get_yticklabels(), fontproperties=tick_font)

                # enlarge SHAP's colorbar labels ("Feature value" + High/Low) to match the axis fonts
                for _cbar_ax in plt.gcf().axes:
                    if _cbar_ax is not ax:
                        _cbar_ax.yaxis.label.set_size(40)
                        _cbar_ax.tick_params(labelsize=35)

                # save
                plt.savefig(
                    output_path
                    + f"{'final/' if isfinal else ''}feature_importances/{name}_shap_beeplot.pdf",
                    bbox_inches="tight",
                    # opaque background + 300dpi: SHAP's beeswarm otherwise rasterizes with a
                    # transparent soft-mask, which breaks CMYK print.
                    dpi=300,
                    facecolor="white",
                    transparent=False,
                )
                pdf.savefig(bbox_inches="tight")
                plt.close()

                # create figure first
                plt.figure(figsize=(70, 30))

                # draw shap beeswarm plot
                shap.plots.beeswarm(shap_full, show=False, max_display=70)
                plt.gcf().set_size_inches(40, 60)
                # grab the current axis
                ax = plt.gca()

                tick_font = font_manager.FontProperties(weight="normal", size=35)

                # set labels and tick params
                ax.set_xlabel("Mean SHAP Value", fontsize=40, fontweight="normal")
                ax.set_ylabel("Feature", fontsize=40, fontweight="normal")
                ax.tick_params(axis="both", which="major", labelsize=35)
                ax.set_xticks(np.arange(-1, 1.1, 0.5))
                ax.set_xlim(-1, 1)
                plt.setp(ax.get_xticklabels(), fontproperties=tick_font)
                plt.setp(ax.get_yticklabels(), fontproperties=tick_font)

                # enlarge SHAP's colorbar labels ("Feature value" + High/Low) to match the axis fonts
                for _cbar_ax in plt.gcf().axes:
                    if _cbar_ax is not ax:
                        _cbar_ax.yaxis.label.set_size(40)
                        _cbar_ax.tick_params(labelsize=35)

                os.makedirs(
                    output_path
                    + f"{'final/' if isfinal else ''}feature_importances/fullshap/",
                    exist_ok=True,
                )
                # save
                plt.savefig(
                    output_path
                    + f"{'final/' if isfinal else ''}feature_importances/fullshap/{name}_shap_beeplot.pdf",
                    bbox_inches="tight",
                    dpi=300,  # see note on the top-15 beeplot save above (print-safety kwargs)
                    facecolor="white",
                    transparent=False,
                )
                pdf.savefig(bbox_inches="tight")
                plt.close()

                # create figure first
                plt.figure(figsize=(70, 30))

                # draw shap beeswarm plot
                shap.plots.beeswarm(shap_1, show=False, max_display=70)
                plt.gcf().set_size_inches(40, 60)
                # grab the current axis
                ax = plt.gca()

                tick_font = font_manager.FontProperties(weight="normal", size=35)

                # set labels and tick params
                ax.set_xlabel("Mean SHAP Value", fontsize=40, fontweight="normal")
                ax.set_ylabel("Feature", fontsize=40, fontweight="normal")
                ax.tick_params(axis="both", which="major", labelsize=35)
                ax.set_xticks(np.arange(-1, 1.1, 0.5))
                ax.set_xlim(-1, 1)
                plt.setp(ax.get_xticklabels(), fontproperties=tick_font)
                plt.setp(ax.get_yticklabels(), fontproperties=tick_font)

                # enlarge SHAP's colorbar labels ("Feature value" + High/Low) to match the axis fonts
                for _cbar_ax in plt.gcf().axes:
                    if _cbar_ax is not ax:
                        _cbar_ax.yaxis.label.set_size(40)
                        _cbar_ax.tick_params(labelsize=35)

                os.makedirs(
                    output_path
                    + f"{'final/' if isfinal else ''}feature_importances/fullshap/",
                    exist_ok=True,
                )
                # save
                plt.savefig(
                    output_path
                    + f"{'final/' if isfinal else ''}feature_importances/fullshap/{name}_shap_beeplot_1.pdf",
                    bbox_inches="tight",
                    dpi=300,  # see note on the top-15 beeplot save above (print-safety kwargs)
                    facecolor="white",
                    transparent=False,
                )
                pdf.savefig(bbox_inches="tight")
                plt.close()
                plt.rcParams["text.usetex"] = _prev_usetex_shap

                # save shap values to csv
                shap_df.to_csv(
                    output_path
                    + f"{'final/' if isfinal else ''}feature_importances/{name}_shap_values.csv",
                    index=False,
                )

                if shap_values_per_fold is not None:
                    waterfall_output = (
                        output_path
                        + f"{'final/' if isfinal else ''}feature_importances/waterfall/{name}/"
                    )
                    os.makedirs(waterfall_output, exist_ok=True)

                    _config_has_pins = any(k[0] == name for k in PINNED_WATERFALLS)
                    for variant_idx, fold_exps in enumerate(shap_values_per_fold):
                        for fold_idx, fold_exp in enumerate(fold_exps):
                            print(
                                f"Waterfall plots for variant {variant_idx}, fold {fold_idx}"
                            )
                            # If this config has manuscript pins, only generate the
                            # pinned (variant, fold) panels; the rest are unused.
                            if (
                                _config_has_pins
                                and (name, variant_idx, fold_idx)
                                not in PINNED_WATERFALLS
                            ):
                                continue
                            y_true = (
                                np.asarray(y_tests_per_fold[variant_idx][fold_idx])
                                .astype(np.int32)
                                .ravel()
                            )
                            y_prob = (
                                np.asarray(y_probas_per_fold[variant_idx][fold_idx])
                                .astype(np.float64)
                                .ravel()
                            )

                            shap_values_data = fold_exp.values
                            feature_data = fold_exp.data

                            values_df = pd.DataFrame(shap_values_data)
                            nan_in_values_mask = (
                                values_df.isnull().any(axis=1).to_numpy()
                            )

                            feature_df = pd.DataFrame(feature_data)
                            nan_in_data_mask = (
                                feature_df.isnull().any(axis=1).to_numpy()
                            )

                            final_missing_mask = np.logical_or(
                                nan_in_values_mask, nan_in_data_mask
                            )
                            valid_indices = np.where(~final_missing_mask)[0]

                            fpr, _, thresholds = roc_curve(y_true, y_prob)
                            base_fpr = np.linspace(0, 1, 101)
                            interp_thresholds = np.interp(base_fpr, fpr, thresholds)

                            idx = 15
                            new_threshold = interp_thresholds[idx]

                            y_pred = (y_prob >= new_threshold).astype(int)

                            y_true_valid = y_true[valid_indices]
                            y_pred_valid = y_pred[valid_indices]
                            tp_indices_r = np.where(
                                (y_true_valid == 1) & (y_pred_valid == 1)
                            )[0]
                            fn_indices_r = np.where(
                                (y_true_valid == 1) & (y_pred_valid == 0)
                            )[0]
                            fp_indices_r = np.where(
                                (y_true_valid == 0) & (y_pred_valid == 1)
                            )[0]
                            tn_indices_r = np.where(
                                (y_true_valid == 0) & (y_pred_valid == 0)
                            )[0]

                            tp_indices = valid_indices[tp_indices_r]
                            fn_indices = valid_indices[fn_indices_r]
                            fp_indices = valid_indices[fp_indices_r]
                            tn_indices = valid_indices[tn_indices_r]

                            pin = PINNED_WATERFALLS.get((name, variant_idx, fold_idx))
                            if pin is not None:
                                # Manuscript figures: plot fixed, deterministic
                                # X_test row positions instead of a random draw.
                                indices_to_plot = {
                                    desc: np.asarray(pos, dtype=int)
                                    for desc, pos in pin.items()
                                }
                            else:
                                indices_to_plot = {
                                    "Correctly Classified as AD": np.random.choice(
                                        tp_indices,
                                        size=min(2, len(tp_indices)),
                                        replace=False,
                                    ),
                                    "Misclassified AD as CN": np.random.choice(
                                        fn_indices,
                                        size=min(2, len(fn_indices)),
                                        replace=False,
                                    ),
                                    "Misclassified CN as AD": np.random.choice(
                                        fp_indices,
                                        size=min(2, len(fp_indices)),
                                        replace=False,
                                    ),
                                    "Correctly Classified as CN": np.random.choice(
                                        tn_indices,
                                        size=min(2, len(tn_indices)),
                                        replace=False,
                                    ),
                                }

                            for description, indices in indices_to_plot.items():
                                if len(indices) > 0:
                                    print(
                                        f"\n--- Plotting for: variant {variant_idx} fold {fold_idx} {description} ---"
                                    )
                                    for i, sample_idx in enumerate(indices):
                                        print(
                                            f"  - Plotting sample {i + 1} (Fold Test Index: {sample_idx})"
                                        )
                                        _prev_usetex_wf = plt.rcParams.get(
                                            "text.usetex", False
                                        )
                                        plt.rcParams["text.usetex"] = False
                                        shap.plots.waterfall(
                                            fold_exp[sample_idx],
                                            max_display=16,
                                            show=False,
                                        )
                                        plt.gcf().set_size_inches(15, 10)
                                        ax = plt.gca()

                                        original_labels = ax.get_yticklabels()
                                        masked_labels = [
                                            mask_feature_value(label.get_text())
                                            for label in original_labels
                                        ]
                                        ax.set_yticklabels(masked_labels)

                                        plt.tight_layout()
                                        plt.savefig(
                                            waterfall_output
                                            + f"{name} - {description} - variant_{variant_idx}_fold_{fold_idx}_sample_{i + 1}.pdf"
                                        )
                                        plt.close()
                                        plt.rcParams["text.usetex"] = _prev_usetex_wf
                                else:
                                    print(
                                        f"\n--- No samples found for: variant {variant_idx} fold {fold_idx} {description} ---"
                                    )


def process_ad_vs_healthy(
    input_dfs,
    name,
    num_features,
    cat_features,
    is_final=True,
    is_shapley=True,
    is_csv=False,
    compute_sampling=False,
    output_path="./results/first_data/",
):
    all_metrics = defaultdict(list)
    all_preds, all_probas = [], []
    all_pred_weights, all_probas_weights = [], []
    all_tests = []
    all_importances = []
    all_shap_values = []
    all_test_eids = []
    all_shap_values_per_fold = []
    all_y_tests_per_fold = []
    all_y_probas_per_fold = []

    for idx, data in enumerate(input_dfs):
        # Filter out 'dementia' class
        df = data.copy()
        filter = df["Status"] == "dementia"
        df.drop(df[filter].index, inplace=True)
        # Map status to binary classification
        df["Status"] = df["Status"].map({"healthy": 0, "AD": 1})

        result = make_prediction_2class(
            df,
            cat_features=cat_features,
            num_features=num_features,
            name=f"{name} - AD vs Healthy",
            isfinal=is_final,
            target="AD",
            isShapley=is_shapley,
            output_path=output_path,
            is_csv=is_csv,
            random_state=41 + idx,
        )

        # Concatenate fold predictions into OOF arrays (reused for both metrics and visualization)
        y_true_oof = np.concatenate([np.asarray(yt) for yt in result["y_tests"]])
        y_pred_oof = np.concatenate(result["y_preds"])
        y_proba_oof = np.concatenate(result["y_probas"])
        y_pred_w_oof = np.concatenate(result["y_pred_weights"])
        y_proba_w_oof = np.concatenate(result["y_probas_weights"])

        # Train metrics: keep fold average (no OOF equivalent for training data)
        for k, v in result["metrics"].items():
            if "train" in k:
                all_metrics[k].append(np.mean(v))

        # Test metrics: compute from OOF predictions
        all_metrics["roc_auc_test_weighted"].append(
            roc_auc_score(y_true_oof, y_proba_w_oof, average="macro")
        )
        all_metrics["roc_auc_test_unweighted"].append(
            roc_auc_score(y_true_oof, y_proba_oof, average="macro")
        )
        all_metrics["mcc_test_weighted"].append(
            matthews_corrcoef(y_true_oof, y_pred_w_oof)
        )
        all_metrics["mcc_test_unweighted"].append(
            matthews_corrcoef(y_true_oof, y_pred_oof)
        )
        all_metrics["f1_test_weighted"].append(
            f1_score(y_true_oof, y_pred_w_oof, average="macro", zero_division=1)
        )
        all_metrics["f1_test_unweighted"].append(
            f1_score(y_true_oof, y_pred_oof, average="macro", zero_division=1)
        )
        all_metrics["accuracy_test_weighted"].append(
            accuracy_score(y_true_oof, y_pred_w_oof)
        )
        all_metrics["accuracy_test_unweighted"].append(
            accuracy_score(y_true_oof, y_pred_oof)
        )
        all_metrics["auprc_test_weighted"].append(
            average_precision_score(y_true_oof, y_proba_w_oof)
        )
        all_metrics["auprc_test_unweighted"].append(
            average_precision_score(y_true_oof, y_proba_oof)
        )
        sens_w, spec_w = sensitivity_specificity(y_true_oof, y_pred_w_oof)
        sens_u, spec_u = sensitivity_specificity(y_true_oof, y_pred_oof)
        all_metrics["sensitivity_test_weighted"].append(sens_w)
        all_metrics["sensitivity_test_unweighted"].append(sens_u)
        all_metrics["specificity_test_weighted"].append(spec_w)
        all_metrics["specificity_test_unweighted"].append(spec_u)

        # Store OOF arrays for visualization
        all_preds.append(y_pred_oof)
        all_probas.append(y_proba_oof)
        all_pred_weights.append(y_pred_w_oof)
        all_probas_weights.append(y_proba_w_oof)
        all_tests.append(y_true_oof)

        # Concatenate fold eids into a single OOF eid array
        eids_oof = sum(result["test_eids"], [])
        all_test_eids.append(eids_oof)

        # Average feature importances across folds
        all_importances.append(np.mean(result["feature_importances"], axis=0))
        if is_shapley:
            # Combine fold SHAP values within this dataset
            combined = result["shap_values"]
            all_shap_values.append(
                shap.Explanation(
                    values=np.vstack([sv.values for sv in combined]),
                    base_values=np.concatenate([sv.base_values for sv in combined]),
                    data=np.vstack([sv.data for sv in combined]),
                    feature_names=combined[0].feature_names,
                )
            )
            # Keep per-fold structure so waterfall plots can use each fold's own base_value
            all_shap_values_per_fold.append(combined)
            all_y_tests_per_fold.append(result["y_tests"])
            all_y_probas_per_fold.append(result["y_probas"])

    visualize_combined_results(
        metrics=all_metrics,
        y_preds=all_preds,
        y_probas=all_probas,
        y_pred_weights=all_pred_weights,
        y_probas_weights=all_probas_weights,
        y_tests=all_tests,
        feature_importances=all_importances,
        shap_values=all_shap_values,
        feature_names=result["feature_names"],
        output_path=output_path,
        name=f"{name} - AD vs Healthy",
        target_names=["Healthy", "AD"],
        isfinal=is_final,
        test_eids=all_test_eids,
        shap_values_per_fold=all_shap_values_per_fold if is_shapley else None,
        y_tests_per_fold=all_y_tests_per_fold if is_shapley else None,
        y_probas_per_fold=all_y_probas_per_fold if is_shapley else None,
    )


def process_dementia_vs_healthy(
    input_dfs,
    name,
    num_features,
    cat_features,
    is_final=True,
    is_shapley=True,
    is_csv=False,
    compute_sampling=False,
    output_path="./results/first_data/",
):
    all_metrics = defaultdict(list)
    all_preds, all_probas = [], []
    all_pred_weights, all_probas_weights = [], []
    all_tests = []
    all_importances = []
    all_shap_values = []
    all_test_eids = []
    all_shap_values_per_fold = []
    all_y_tests_per_fold = []
    all_y_probas_per_fold = []

    for idx, data in enumerate(input_dfs):
        df = data.copy()
        df["Status"] = df["Status"].map({"healthy": 0, "dementia": 1, "AD": 1})

        result = make_prediction_2class(
            df,
            cat_features=cat_features,
            num_features=num_features,
            name=f"{name} - Dementia vs Healthy",
            isfinal=is_final,
            target="dementia",
            isShapley=is_shapley,
            output_path=output_path,
            is_csv=is_csv,
            random_state=41 + idx,
        )

        # Concatenate fold predictions into OOF arrays (reused for both metrics and visualization)
        y_true_oof = np.concatenate([np.asarray(yt) for yt in result["y_tests"]])
        y_pred_oof = np.concatenate(result["y_preds"])
        y_proba_oof = np.concatenate(result["y_probas"])
        y_pred_w_oof = np.concatenate(result["y_pred_weights"])
        y_proba_w_oof = np.concatenate(result["y_probas_weights"])

        # Train metrics: keep fold average (no OOF equivalent for training data)
        for k, v in result["metrics"].items():
            if "train" in k:
                all_metrics[k].append(np.mean(v))

        # Test metrics: compute from OOF predictions
        all_metrics["roc_auc_test_weighted"].append(
            roc_auc_score(y_true_oof, y_proba_w_oof, average="macro")
        )
        all_metrics["roc_auc_test_unweighted"].append(
            roc_auc_score(y_true_oof, y_proba_oof, average="macro")
        )
        all_metrics["mcc_test_weighted"].append(
            matthews_corrcoef(y_true_oof, y_pred_w_oof)
        )
        all_metrics["mcc_test_unweighted"].append(
            matthews_corrcoef(y_true_oof, y_pred_oof)
        )
        all_metrics["f1_test_weighted"].append(
            f1_score(y_true_oof, y_pred_w_oof, average="macro", zero_division=1)
        )
        all_metrics["f1_test_unweighted"].append(
            f1_score(y_true_oof, y_pred_oof, average="macro", zero_division=1)
        )
        all_metrics["accuracy_test_weighted"].append(
            accuracy_score(y_true_oof, y_pred_w_oof)
        )
        all_metrics["accuracy_test_unweighted"].append(
            accuracy_score(y_true_oof, y_pred_oof)
        )
        all_metrics["auprc_test_weighted"].append(
            average_precision_score(y_true_oof, y_proba_w_oof)
        )
        all_metrics["auprc_test_unweighted"].append(
            average_precision_score(y_true_oof, y_proba_oof)
        )
        sens_w, spec_w = sensitivity_specificity(y_true_oof, y_pred_w_oof)
        sens_u, spec_u = sensitivity_specificity(y_true_oof, y_pred_oof)
        all_metrics["sensitivity_test_weighted"].append(sens_w)
        all_metrics["sensitivity_test_unweighted"].append(sens_u)
        all_metrics["specificity_test_weighted"].append(spec_w)
        all_metrics["specificity_test_unweighted"].append(spec_u)

        # Store OOF arrays for visualization
        all_preds.append(y_pred_oof)
        all_probas.append(y_proba_oof)
        all_pred_weights.append(y_pred_w_oof)
        all_probas_weights.append(y_proba_w_oof)
        all_tests.append(y_true_oof)

        # Concatenate fold eids into a single OOF eid array
        eids_oof = sum(result["test_eids"], [])
        all_test_eids.append(eids_oof)

        # Average feature importances across folds
        all_importances.append(np.mean(result["feature_importances"], axis=0))
        if is_shapley:
            # Combine fold SHAP values within this dataset
            combined = result["shap_values"]
            all_shap_values.append(
                shap.Explanation(
                    values=np.vstack([sv.values for sv in combined]),
                    base_values=np.concatenate([sv.base_values for sv in combined]),
                    data=np.vstack([sv.data for sv in combined]),
                    feature_names=combined[0].feature_names,
                )
            )
            # Keep per-fold structure so waterfall plots can use each fold's own base_value
            all_shap_values_per_fold.append(combined)
            all_y_tests_per_fold.append(result["y_tests"])
            all_y_probas_per_fold.append(result["y_probas"])

    visualize_combined_results(
        metrics=all_metrics,
        y_preds=all_preds,
        y_probas=all_probas,
        y_pred_weights=all_pred_weights,
        y_probas_weights=all_probas_weights,
        y_tests=all_tests,
        feature_importances=all_importances,
        shap_values=all_shap_values,
        feature_names=result["feature_names"],
        output_path=output_path,
        name=f"{name} - Dementia vs Healthy",
        target_names=["Healthy", "Dementia"],
        isfinal=is_final,
        test_eids=all_test_eids,
        shap_values_per_fold=all_shap_values_per_fold if is_shapley else None,
        y_tests_per_fold=all_y_tests_per_fold if is_shapley else None,
        y_probas_per_fold=all_y_probas_per_fold if is_shapley else None,
    )


def balance_healthy_by_min_not_healthy(dataframes, seed=42):
    # make all dataframes the same size by undersampling healthy class

    balanced_dfs = []

    # split into healthy and not healthy for all dataframes
    healthy_groups = [df[df["Status"] == "healthy"].copy() for df in dataframes]
    not_healthy_groups = [
        df[df["Status"].isin(["AD", "dementia"])].copy() for df in dataframes
    ]

    # find minimum size in each group

    min_patient = min(len(f) for f in dataframes)
    rows_need_to_drop = [len(df) - min_patient for df in dataframes]
    print(rows_need_to_drop)

    for i, rdf in enumerate(rows_need_to_drop):
        if rdf > 0:
            healthy_groups[i] = healthy_groups[i].sample(
                n=len(healthy_groups[i]) - rdf, random_state=seed
            )

    # combine and shuffle (preserve eid index for participant tracking)
    for h_df, nh_df in zip(healthy_groups, not_healthy_groups):
        combined = pd.concat([h_df, nh_df])
        balanced_dfs.append(combined.sample(frac=1, random_state=seed))  # shuffle rows

    print(f"Balanced DataFrames: {[len(df) for df in balanced_dfs]}")
    return balanced_dfs
