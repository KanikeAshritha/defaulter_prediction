# bankloan_ml_pipeline.py
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import json
from evidently import Report
from evidently.presets import *
from evidently.metrics import *


def perform_eda(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_features = ['Age', 'Income', 'CCAvg'] 
    categorical_features = ['Family', 'Education', 'Securities Account', 'CD Account',
                            'Online', 'CreditCard', 'HasMortgage']
    target = 'Personal Loan'

    print("\nUnivariate Analysis")

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_features + categorical_features + [target], 1):
        plt.subplot(4, 3, i)
        if col in categorical_features + [target]:
            sns.countplot(x=col, data=df, palette='Set2')
        else:
            sns.histplot(df[col], bins=30, kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    print("\nBivariate Analysis: Numeric vs Target")

    # Bivariate - KDE plots
    for col in numeric_features:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=df, x=col, hue=target, fill=True, common_norm=False, palette='Set2')
        plt.title(f'{col} Distribution by {target}')
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

    print("\nBoxplot: Income vs Target")
    sns.boxplot(data=df, x=target, y='Income')
    plt.title("Income vs Personal Loan")
    plt.show()

    print("\nCorrelation Matrix")
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Experience'] >= 0]

    for col in ['Income', 'CCAvg', 'Mortgage']:
        df[col] = np.log1p(df[col])

    df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)

    df.drop(columns=['Experience', 'Mortgage'], inplace=True)

    return df

def split_data(df, target_col, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def create_rf_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

def create_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='liblinear'))
    ])

def perform_grid_search(pipe, X_train, y_train, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1):
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train_model():
    TARGET_COL = 'Personal Loan'

    RF_PARAM_GRID = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }

    LOGISTIC_PARAM_GRID = {
        'lr__C': [0.01, 0.1, 1, 10],
        'lr__penalty': ['l1', 'l2']
    }

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

    models_config = {
        "RandomForest": {
            "pipeline": create_rf_pipeline(),
            "param_grid": RF_PARAM_GRID
        },
        "LogisticRegression": {
            "pipeline": create_logistic_pipeline(),
            "param_grid": LOGISTIC_PARAM_GRID
        }
    }

    return X_train, X_test, y_train, y_test,models_config


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def log_and_register_model(model, model_name, X_test, y_test):
    evaluate_model(model, X_test, y_test)

    run = mlflow.active_run()
    run_id = run.info.run_id
    experiment = mlflow.get_experiment(run.info.experiment_id)
    artifact_path = f"{model_name}_model"

    mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered_model = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    version = registered_model.version
    unique_model_name = f"{model_name}_v{version}"

    joblib.dump(model, f"{unique_model_name}_pipeline.pkl")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )

    return {
        "unique_model_name": unique_model_name,
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "experiment_name": experiment.name
    }



def mlflow_code(X_train, X_test, y_train, y_test , models_config):
    best_model = None
    best_auc = 0.0
    best_model_info = {}

    for model_name, config in models_config.items():
        with mlflow.start_run(run_name=f"{model_name}_Classifier", nested=True):
            grid_search = perform_grid_search(
                config["pipeline"],
                X_train, y_train,
                config["param_grid"]
            )

            artifact_path = f"{model_name}_model"
            mlflow.sklearn.log_model(sk_model=grid_search.best_estimator_, artifact_path=artifact_path)

            mlflow.log_params(grid_search.best_params_)

            y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("Test AUC", test_auc)

            print(f"\n Model: {model_name} - Test AUC: {test_auc:.4f}")

            if test_auc > best_auc:
                best_auc = test_auc
                best_model = grid_search.best_estimator_
                best_model_info = log_and_register_model(best_model, model_name, X_test, y_test)
            # else:
            #     artifact_path = f"{model_name}_model"
            #     mlflow.sklearn.log_model(sk_model=grid_search.best_estimator_, artifact_path=artifact_path)

    if best_model_info:
        print("\nFINAL BEST MODEL SUMMARY")
        print(f"Model Name         : {best_model_info['model_name']}")
        print(f"Unique Model Name  : {best_model_info['unique_model_name']}")
        print(f"Version            : {best_model_info['version']}")
        print(f"Run ID             : {best_model_info['run_id']}")
        print(f"Experiment Name    : {best_model_info['experiment_name']}")
    else:
        print("\n No model beat baseline AUC. No model registered.")


def clean_metric_name(name):
    return re.sub(r"[^\w\-/\. ]", "_", name)

def log_evidently_reports(train_df, test_df, new_df):
    report_pairs = [
        ("train_vs_test", train_df, test_df),
        ("historical_vs_new", df, new_df)
    ]

    report_configs = [
        ("drift", DataDriftPreset(method='psi')),
        ("summary", DataSummaryPreset())
    ]

    for name, ref_df, curr_df in report_pairs:
        for report_type, preset in report_configs:
            report = Report([preset], include_tests=True)
            result = report.run(reference_data=ref_df, current_data=curr_df)

            html_path = f"evidently_{name}_{report_type}.html"
            result.save_html(html_path)
            mlflow.log_artifact(html_path)

            json_data = json.loads(result.json())
            # print(f"\n{report_type.capitalize()} Metrics for: {name}")
            for metric in json_data.get("metrics", []):
                metric_id = metric.get("metric_id") or metric.get("metric", "")
                value = metric.get("value", None)

                if isinstance(value, dict):
                    for sub_name, sub_val in value.items():
                        if isinstance(sub_val, (int, float)):
                            metric_name = clean_metric_name(f"{name}_{report_type}_{metric_id}_{sub_name}")
                            # print(f"{metric_name}: {sub_val}")
                            mlflow.log_metric(metric_name, sub_val)
                elif isinstance(value, (int, float)):
                    metric_name = clean_metric_name(f"{name}_{report_type}_{metric_id}")
                    # print(f"{metric_name}: {value}")
                    mlflow.log_metric(metric_name, value)
                elif "ValueDrift(column=" in metric_id:
                    try:
                        col_name = metric_id.split("ValueDrift(column=")[1].split(",")[0]
                        metric_name = clean_metric_name(f"{name}_{report_type}_{col_name}")
                        # print(f"{metric_name}: {value:.6f}")
                        mlflow.log_metric(metric_name, value)
                    except Exception as e:
                        print(f"Error parsing {report_type} metric: {metric} -> {e}")

if __name__ == "__main__":
    mlflow.set_experiment("BankLoan_ML_Tracking")

    df = preprocess_data("Bank_Personal_Loan_Modelling.csv")
    new_data = preprocess_data("New Customer Bank_Personal_Loan.csv") 

    X_train, X_test, y_train, y_test , models_config = train_model()

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)


    flow = mlflow_code(X_train, X_test, y_train, y_test, models_config)

# Start separate run for logging Evidently reports
    with mlflow.start_run():
        log_evidently_reports(train_df, test_df, new_data)


    
