import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import mlflow
import dagshub
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

iris = load_iris()

def create_model_signature(X_test, y_pred):
    """Create MLflow model signature"""
    input_schema = Schema([
        ColSpec("double", "sepal length (cm)"),
        ColSpec("double", "sepal width (cm)"),
        ColSpec("double", "petal length (cm)"),
        ColSpec("double", "petal width (cm)")
    ])
    output_schema = Schema([ColSpec("long", "target")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)

from prefect import task, flow

@task(name="preprocess_data")
def preprocess(data):
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    return X_train, X_test, y_train, y_test

@task(name="model_selection")
def model_selection(models, X_train, y_train, X_test, y_test):
    mlflow.set_tracking_uri("https://dagshub.com/varaprasad7654321/Test.mlflow")
    dagshub.init(repo_owner='varaprasad7654321', repo_name='Test', mlflow=True)
    mlflow.set_experiment("MLFlow_test")
    
    model_scores = {}
    for model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Model score for {model.__class__.__name__} is {score}")
        model_scores[model.__class__.__name__] = score

        y_pred = model.predict(X_test)
        signature = create_model_signature(X_test, y_pred)

        with mlflow.start_run(run_name=model.__class__.__name__) as run:
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_test.iloc[:5]
            )
            mlflow.log_metric("accuracy", score)
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("f1", f1_score(y_test, y_pred, average='weighted'))
    
    return model_scores

@flow(name="ml_training_pipeline")
def main():
    X_train, X_test, y_train, y_test = preprocess(iris)
    models = [
        LogisticRegression(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        SVC()
    ]
    model_scores = model_selection(models, X_train, y_train, X_test, y_test)
    print("\nFinal Model Scores:")
    for model_name, score in model_scores.items():
        print(f"{model_name}: {score:.4f}")

if __name__ == "__main__":
    main()
