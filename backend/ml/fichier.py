from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "iris_model.pkl"

# MLflow experiment
mlflow.set_experiment("iris_experiment")


def train_and_save():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()

    # Start an MLflow run
    with mlflow.start_run():
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        # Save the model as .pkl for FastAPI / Docker / Azure
        joblib.dump(lr, MODEL_PATH)
        print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
