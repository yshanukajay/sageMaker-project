
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)
import sklearn
import joblib
import argparse
import os
import pandas as pd

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


if __name__ == "__main__":

    print("[info] Extracting arguments.")
    parser=argparse.ArgumentParser()

    ## hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    ## data, model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")

    args, _ = parser.parse_known_args()

    print("Scikit-learn version:", sklearn.__version__)
    print("Joblib version:", joblib.__version__)

    print("[info] Reading training and test data.")
    print()

    train_data = pd.read_csv(os.path.join(args.train, args.train_file))
    test_data = pd.read_csv(os.path.join(args.test, args.test_file))

    features = train_data.columns[:-1]
    label = train_data.columns[-1]

    print("Building training and test datasets.")
    print()
    X_train = train_data[features]
    y_train = train_data[label]
    X_test = test_data[features]
    y_test = test_data[label]

    print("Column order")
    print()

    print(X_train.columns)
    print(X_test.columns)
    print()

    print("Label column is:", label)
    print()

    print("Data shapes:")
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    print()

    print("[info] Training model.")
    print()

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        verbose=1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model saved to:", model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")

    print("Test Accuracy:", acc)
    print("Test Precision:", precision)
    print("Classification Report:")
    print(test_report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print()
