import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from prefect import task, flow
import mlflow
import dagshub
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models import infer_signature

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_diabetes, load_wine, load_breast_cancer

iris = load_iris()

@task
def create_model_signature(X_train, y_pred):
    """Create MLflow model signature"""
    # Define input schema
    input_schema = Schema([
        ColSpec("double", "sepal length (cm)"),
        ColSpec("double", "sepal width (cm)"),
        ColSpec("double", "petal length (cm)"),
        ColSpec("double", "petal width (cm)")
    ])

    # Define output schema
    output_schema = Schema([ColSpec("long", "target")])

    # Create signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    return signature

@task
def preprocess(data):
    X = pd.DataFrame(data.data, columns = data.feature_names)
    y = pd.Series(data.target, name = 'target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)

    return X_train, X_test, y_train, y_test

def model_selection(models, X_train, y_train, X_test, y_test):
    mlflow.set_tracking_uri("https://dagshub.com/varaprasad7654321/Test.mlflow")
    dagshub.init(repo_owner='varaprasad7654321', repo_name='Test', mlflow=True)
    mlflow.set_experiment("MLFlow_test")
    model_score = {}
    for i in models:
        i.fit(X_train, y_train)
        score = i.score(X_test, y_test)

        print(f"Model score for {i} is {score}")
        model_score[i] = score

        y_pred = i.predict(X_test)

        signature = create_model_signature(X_test, y_pred)

        with mlflow.start_run(run_name=str(i)) as run:
            mlflow.sklearn.log_model(
                            i,
                            "model",
                            signature=signature,
                            input_example=X_test.iloc[:5]  # Log sample input
                        )
            mlflow.log_metric("accuracy", i.score(X_test, y_test))
            mlflow.log_metric("precision", precision_score(y_test, i.predict(X_test),average='weighted'))
            mlflow.log_metric("recall", recall_score(y_test, i.predict(X_test),average='weighted'))
            mlflow.log_metric("f1", f1_score(y_test, i.predict(X_test),average='weighted'))


@flow(log_prints = True)
def work():
    X_train, X_test, y_train, y_test = preprocess(iris)
    models = [ LogisticRegression(), RandomForestClassifier(),  DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]
    model_score = model_selection(models, X_train, y_train, X_test, y_test)
    print(model_score)

if __name__ == "__main__":
    work()
