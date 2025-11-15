import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from algorithms import PCA, SoftmaxRegression, XGBoostClassifier, KNN
import pandas as pd
import argparse

def create_arrays(df):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y


def filter_pair(X, y, a, b):
    mask = (y == a) | (y == b)
    X_ab = X[mask]
    y_ab = (y[mask] == b).astype(int)
    return X_ab, y_ab

def apply_pairwise_corrections(x, base_pred, pairwise_models, confusing_pairs):
    for (a, b) in confusing_pairs:
        if base_pred == a or base_pred == b:
            model = pairwise_models[(a, b)]
            out = model.predict(np.array([x]))[0]
            return b if out == 1 else a
    return base_pred

def run_pipeline(X_train, y_train, X_val, y_val):
    pca = PCA(k=60)
    X_train_pca = pca.fit(X_train)
    X_val_pca = pca.transform(X_val)
    softmax = SoftmaxRegression(lr=0.3, num_iters=3000, reg=0.01)
    softmax.fit(X_train_pca, y_train)

    soft_pred = softmax.predict(X_val_pca)
    soft_proba = softmax._predict_proba(X_val_pca)

    confusing_pairs = [
        (9, 7),
        (3, 5),
        (8, 3),
        (8, 5),
        (2, 6),
        (5, 6),
        (8, 1),
        (9, 4),
        (3, 7)
    ]

    pairwise_models = {}

    for (a, b) in confusing_pairs:
        X_ab, y_ab = filter_pair(X_train_pca, y_train, a, b)

        clf = XGBoostClassifier(
            n_estimators=150,
            learning_rate=0.2,
            max_depth=3,
            reg_lambda=0.1,
            colsample=0.3
        )
        clf.fit(X_ab, y_ab)
        pairwise_models[(a, b)] = clf

    corrected_softmax = []
    for i in range(len(X_val)):
        base = soft_pred[i]
        corr = apply_pairwise_corrections(
            X_val_pca[i],
            base,
            pairwise_models,
            confusing_pairs
        )
        corrected_softmax.append(corr)

    corrected_softmax = np.array(corrected_softmax)

    knn = KNN(k=3)
    knn.fit(X_train_pca, y_train)
    knn_pred = knn.predict(X_val_pca)
    final_preds = []

    for i in range(len(y_val)):
        s = soft_pred[i]
        cs = corrected_softmax[i]
        k = knn_pred[i]
        s_conf = np.max(soft_proba[i])

        if s != cs:
            final_preds.append(k)
            continue

        if s_conf < 0.85:
            final_preds.append(k)
            continue

        if (cs % 2) != (k % 2):
            final_preds.append(k)
            continue

        final_preds.append(k)

    final_preds = np.array(final_preds)

    print("Accuracy:", accuracy_score(y_val, final_preds))
    print("Weighted F1:", f1_score(y_val, final_preds, average='weighted'))
    print(classification_report(y_val, final_preds))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to train file"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test file"
    )

    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    X_train, y_train = create_arrays(train_df)
    X_val, y_val = create_arrays(val_df)

    run_pipeline(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()