from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("iris-train")

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "iris_model.pkl"

# MLflow experiment
mlflow.set_experiment("iris_experiment")


def train_and_save():
    logger.info("Loading Iris dataset")
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)
    logger.info(f"Dataset loaded: X shape={X.shape}, y shape={y.shape}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(
        f"Data split done: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Enable autologging for scikit-learn
    logger.info("Enabling MLflow autologging for scikit-learn")
    mlflow.sklearn.autolog()

    # Start an MLflow run
    logger.info("Starting MLflow run")
    with mlflow.start_run():
        lr = LogisticRegression(**params)
        logger.info("Fitting model")
        lr.fit(X_train, y_train)

        # Evaluate model
        accuracy = lr.score(X_test, y_test)
        logger.info(f"Test accuracy: {accuracy:.4f}")

        # Save the model as .pkl for FastAPI / Docker / Azure
        logger.info(f"Saving model to {MODEL_PATH}")
        joblib.dump(lr, MODEL_PATH)
        print(f"Model saved at: {MODEL_PATH}")
        logger.info("Training and saving completed")


if __name__ == "__main__":
    train_and_save()
