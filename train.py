"""Breast cancer diagnosis with machine learning.

Trains Logistic Regression and Random Forest on the scikit-learn
breast cancer dataset and prints accuracy, precision, recall and
confusion matrices for both models. Also saves a bar chart of the
top 10 most important features (Random Forest) to outputs/.

For learning and portfolio demonstration only — not clinical or
medical advice.

Usage:
    python src/train.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    print("Dataset shape:", X.shape)
    print("Target labels:", dict(zip(data.target_names, [0, 1])))
    print("Missing values in features:", int(X.isna().sum().sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Train size:", X_train.shape, "| Test size:", X_test.shape)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    for name, model, Xt in [
        ("Logistic Regression", lr, X_test_s),
        ("Random Forest", rf, X_test),
    ]:
        pred = model.predict(Xt)
        print(f"--- {name} ---")
        print("Accuracy :", round(accuracy_score(y_test, pred), 4))
        print("Precision:", round(precision_score(y_test, pred), 4))
        print("Recall   :", round(recall_score(y_test, pred), 4))
        print("Confusion matrix (rows = actual, cols = predicted):")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=data.target_names))

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print("Top 10 most important features (Random Forest):")
    print(top10.round(4))

    os.makedirs("outputs", exist_ok=True)
    top10.sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title("Top 10 most important features (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    print("Saved: outputs/feature_importance.png")


if __name__ == "__main__":
    main()
