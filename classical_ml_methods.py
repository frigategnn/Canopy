from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd

from typing import List, Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


def _accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _prepare_splits(
    augmented_features: np.ndarray, data
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    X = (
        augmented_features
        if isinstance(augmented_features, np.ndarray)
        else np.asarray(augmented_features)
    )
    y = data.y.cpu().numpy() if hasattr(data.y, "cpu") else np.asarray(data.y)

    train_mask = (
        data.train_mask.cpu().numpy()
        if hasattr(data.train_mask, "cpu")
        else np.asarray(data.train_mask)
    ).astype(bool)
    val_mask = (
        data.val_mask.cpu().numpy()
        if hasattr(data.val_mask, "cpu")
        else np.asarray(data.val_mask)
    ).astype(bool)
    test_mask = (
        data.test_mask.cpu().numpy()
        if hasattr(data.test_mask, "cpu")
        else np.asarray(data.test_mask)
    ).astype(bool)

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    # print(X_train.shape, X_val.shape, X_test.shape)
    # exit()
    return X, y, splits


def run_classical_ml(
    augmented_features: np.ndarray,
    data,
    output_csv_path: Union[str, Path] = "classical_ml_results.csv",
) -> pd.DataFrame:
    """Train classical ML models and save accuracies.

    Models (CPU via scikit-learn):
    - Logistic Regression
    - KNN Classifier
    - SVM Classifier (RBF)
    - AdaBoost (Decision Stumps)

    Gradient Boosting via XGBoost with GPU if available, falling back to CPU.

    Uses data.y for labels and masks (train/val/test) from data.
    Features are provided as augmented_features.
    """
    X, y, splits = _prepare_splits(augmented_features, data)

    # Standardize features fit on train only
    scaler = StandardScaler()
    X_train, y_train = splits["train"]
    scaler.fit(X_train)

    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(splits["val"][0])
    X_test_s = scaler.transform(splits["test"][0])

    results = []

    # 1) Logistic Regression
    lr = LogisticRegression(max_iter=2000, tol=1e-4, C=0.01, n_jobs=-1)
    lr.fit(X_train_s, y_train)
    y_tr_pred = lr.predict(X_train_s)
    y_val_pred = lr.predict(X_val_s)
    y_te_pred = lr.predict(X_test_s)
    results.append(
        {
            "model": "LogisticRegression(sklearn)",
            "train_acc": _accuracy_np(y_train, y_tr_pred),
            "val_acc": _accuracy_np(splits["val"][1], y_val_pred),
            "test_acc": _accuracy_np(splits["test"][1], y_te_pred),
        }
    )

    print(pd.DataFrame(results[-1], index=[""]))

    # 2) KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", n_jobs=-1)
    knn.fit(X_train_s, y_train)
    y_tr_pred = knn.predict(X_train_s)
    y_val_pred = knn.predict(X_val_s)
    y_te_pred = knn.predict(X_test_s)
    results.append(
        {
            "model": "KNN(sklearn)",
            "train_acc": _accuracy_np(y_train, y_tr_pred),
            "val_acc": _accuracy_np(splits["val"][1], y_val_pred),
            "test_acc": _accuracy_np(splits["test"][1], y_te_pred),
        }
    )

    print(pd.DataFrame(results[-1], index=[""]))

    # 3) SVM (RBF)
    svm = SVC(C=1.0, kernel="rbf", gamma="scale")
    svm.fit(X_train_s, y_train)
    y_tr_pred = svm.predict(X_train_s)
    y_val_pred = svm.predict(X_val_s)
    y_te_pred = svm.predict(X_test_s)
    results.append(
        {
            "model": "SVM_RBF(sklearn)",
            "train_acc": _accuracy_np(y_train, y_tr_pred),
            "val_acc": _accuracy_np(splits["val"][1], y_val_pred),
            "test_acc": _accuracy_np(splits["test"][1], y_te_pred),
        }
    )

    print(pd.DataFrame(results[-1], index=[""]))

    # 4) AdaBoost
    base_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
    ada = AdaBoostClassifier(estimator=base_tree, n_estimators=200, learning_rate=0.9)
    ada.fit(X_train_s, y_train)
    y_tr_pred = ada.predict(X_train_s)
    y_val_pred = ada.predict(X_val_s)
    y_te_pred = ada.predict(X_test_s)
    results.append(
        {
            "model": "AdaBoost(sklearn)",
            "train_acc": _accuracy_np(y_train, y_tr_pred),
            "val_acc": _accuracy_np(splits["val"][1], y_val_pred),
            "test_acc": _accuracy_np(splits["test"][1], y_te_pred),
        }
    )

    print(pd.DataFrame(results[-1], index=[""]))

    # 5) Gradient Boost (XGBoost - try GPU, fallback to CPU)
    num_classes = int(np.max(y)) + 1
    xgb_params = dict(
        objective="multi:softmax",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.1,
        subsample=0.8,
        reg_lambda=2,
        reg_alpha=0.03,
        gamma=0.1,
        min_child_weight=3,
        max_depth=5,
        verbosity=0,
    )

    # Try GPU first
    try:
        xgb = XGBClassifier(
            tree_method="hist", predictor="gpu_predictor", device="cuda:0", **xgb_params
        )
        xgb.fit(
            splits["train"][0],
            splits["train"][1],
            eval_set=[(splits["val"][0], splits["val"][1])],
            verbose=True,
        )
    except Exception:
        xgb = XGBClassifier(tree_method="hist", predictor="auto", **xgb_params)
        xgb.fit(
            splits["train"][0],
            splits["train"][1],
            eval_set=[(splits["val"][0], splits["val"][1])],
            verbose=True,
        )

    y_tr_pred = xgb.predict(splits["train"][0])
    y_val_pred = xgb.predict(splits["val"][0])
    y_te_pred = xgb.predict(splits["test"][0])
    results.append(
        {
            "model": "GradientBoost(XGBoost)",
            "train_acc": _accuracy_np(splits["train"][1], y_tr_pred),
            "val_acc": _accuracy_np(splits["val"][1], y_val_pred),
            "test_acc": _accuracy_np(splits["test"][1], y_te_pred),
        }
    )

    print(pd.DataFrame(results[-1], index=[""]))

    df = pd.DataFrame(results, columns=["model", "train_acc", "val_acc", "test_acc"])
    df.to_csv(output_csv_path, index=False)
    return df
