
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

mlflow.set_experiment("BankLoan_ML_Tracking")


# %%


# %%
df = pd.read_csv(r'Bank_Personal_Loan_Modelling.csv')
df.head()


# %%

# Select only numerical columns
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Generate descriptive statistics
summary_stats = numerical_df.describe().T  # Transposed for easier viewing

# Add IQR and Range
summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
summary_stats['Range'] = summary_stats['max'] - summary_stats['min']

# Display the result
summary_stats


# %%
# Count negatives before removal
print("Negative experience count before removal:", (df['Experience'] < 0).sum())

# Remove rows with negative experience
df = df[df['Experience'] >= 0]

# Confirm removal
print("Negative experience count after removal:", (df['Experience'] < 0).sum())
print("Updated dataset shape:", df.shape)


# %%
import numpy as np

for col in ['Income', 'CCAvg', 'Mortgage']:
    # plt.figure(figsize=(12,5))
    
    # plt.subplot(1,2,1)
    # sns.histplot(df[col], bins=30, kde=True, color='skyblue')
    # plt.title(f'{col} - Original Distribution')
    
    # plt.subplot(1,2,2)
    # sns.histplot(np.log1p(df[col]), bins=30, kde=True, color='orange')
    # plt.title(f'{col} - Log Transformed Distribution')
    
    # plt.show()

    # Replace original column with log transformed version for modeling
    df[col] = np.log1p(df[col])


# %%
df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)

# sns.countplot(x='HasMortgage', data=df)
# plt.title('Mortgage Ownership Distribution')
# plt.show()


# %%
features_to_plot = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage', 'HasMortgage', 'Personal Loan']

# plt.figure(figsize=(15,10))
# for i, col in enumerate(features_to_plot, 1):
#     plt.subplot(3,3,i)
#     if col == 'Personal Loan' or col == 'HasMortgage':
#         sns.countplot(x=col, data=df, palette='Set2')
#     else:
#         sns.histplot(df[col], bins=30, kde=True)
#     plt.title(col)
# plt.tight_layout()
# plt.show()


# %%

# Select only numerical columns
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Generate descriptive statistics
summary_stats = numerical_df.describe().T  # Transposed for easier viewing

# Add IQR and Range
summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
summary_stats['Range'] = summary_stats['max'] - summary_stats['min']

# Display the result
summary_stats

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Numerical features to analyze
numeric_features = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']

# Categorical/Binary features
categorical_features = ['Family', 'Education', 'Securities Account', 'CD Account', 
                        'Online', 'CreditCard', 'HasMortgage']

# # 1. Numerical features vs. target
# for col in numeric_features:
#     plt.figure(figsize=(8, 4))
#     sns.kdeplot(data=df, x=col, hue='Personal Loan', fill=True, common_norm=False, palette='Set2')
#     plt.title(f'{col} Distribution by Personal Loan')
#     plt.xlabel(col)
#     plt.ylabel("Density")
#     plt.show()


# %%
# sns.boxplot(data=df, x='Personal Loan', y='Income')
# plt.title("Income vs Personal Loan")
# plt.show()


# %%
# Group statistics
group_stats = df.groupby('Personal Loan').mean(numeric_only=True).T
group_stats['Diff'] = group_stats[1] - group_stats[0]
group_stats.sort_values('Diff', ascending=False)


# # %%
# # Correlation matrix for numerical features
# plt.figure(figsize=(10, 8))
# corr = df.corr(numeric_only=True)
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()


# %%


# %%
# Drop highly correlated/redundant features
df.drop(columns=['Experience', 'Mortgage'], inplace=True)

# Confirm changes
print("Remaining columns:", df.columns.tolist())


# # %%
# # Correlation matrix for numerical features
# plt.figure(figsize=(10, 8))
# corr = df.corr(numeric_only=True)
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()


# %%
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def split_data(df, target_col, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

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

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path
    )

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
        stage="Production"
    )

    return {
        "unique_model_name": unique_model_name,
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "experiment_name": experiment.name
    }

if __name__ == "__main__":
    TARGET_COL = 'Personal Loan'
    PARAM_GRID = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

# %%
from sklearn.linear_model import LogisticRegression

def create_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='liblinear')) 
    ])

LOGISTIC_PARAM_GRID = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l1', 'l2']
}

if __name__ == "__main__":
    TARGET_COL = 'Personal Loan'

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

    models_config = {
    "RandomForest": {
        "pipeline": create_pipeline(),
        "param_grid": PARAM_GRID
    },
    "LogisticRegression": {
        "pipeline": create_logistic_pipeline(),
        "param_grid": LOGISTIC_PARAM_GRID
    }
    }

    best_model = None
    best_model_name = None
    best_auc = 0.0
    best_model_info = {}

    for model_name, config in models_config.items():
        with mlflow.start_run(run_name=f"{model_name}_Classifier"):
            grid_search = perform_grid_search(
                config["pipeline"],
                X_train,
                y_train,
                config["param_grid"]
            )
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("Best CV AUC", grid_search.best_score_)

            y_pred = grid_search.best_estimator_.predict(X_test)
            y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("Test AUC", test_auc)

            print(f"\nModel: {model_name} - Test AUC: {test_auc:.4f}")

            if test_auc > best_auc:
                best_auc = test_auc
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                model_info = log_and_register_model(best_model, best_model_name, X_test, y_test)

    print(f"\nBest Model Based on Test AUC: {best_model_name} ({best_auc:.4f})")

    print("\nFINAL BEST MODEL SUMMARY")
    print(f"Model Name         : {model_info['model_name']}")
    print(f"Unique Model Name  : {model_info['unique_model_name']}")
    print(f"Version            : {model_info['version']}")
    print(f"Run ID             : {model_info['run_id']}")
    print(f"Experiment Name    : {model_info['experiment_name']}")



# %%
import joblib
joblib.dump(grid_search.best_estimator_, "model_pipeline.pkl")
print("Saved best model as model_pipeline.pkl")
  


