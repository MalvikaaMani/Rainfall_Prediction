import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint

# Load the dataset
df = pd.read_csv('testset.csv')
print("Dataset loaded successfully.")
print(df.columns)

# --- Data Preprocessing (Expanded Feature Engineering) ---
print("\n--- Data Preprocessing (Expanded Feature Engineering) ---")

df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce')
df['date_only'] = df['datetime_utc'].dt.date
df.dropna(subset=['date_only'], inplace=True)

numerical_features_to_agg = [
    ' _tempm', ' _hum', ' _pressurem', ' _dewptm', ' _vism', ' _wspdm',
    ' _precipm', ' _heatindexm', ' _windchillm'
]
binary_flag_features_to_agg = [' _fog', ' _hail', ' _snow', ' _thunder', ' _tornado']
target_for_aggregation = ' _rain'

all_features_for_agg = numerical_features_to_agg + binary_flag_features_to_agg + [target_for_aggregation]
for col in all_features_for_agg:
    if col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
            print(f"Imputed missing values in '{col}' before aggregation.")
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping imputation for it.")

aggregation_dict = {}
for feature in numerical_features_to_agg:
    if feature in df.columns:
        aggregation_dict[f'{feature.strip()}_mean'] = (feature, 'mean')
        aggregation_dict[f'{feature.strip()}_max'] = (feature, 'max')
        aggregation_dict[f'{feature.strip()}_min'] = (feature, 'min')
        aggregation_dict[f'{feature.strip()}_std'] = (feature, 'std')
    else:
        print(f"Warning: Numerical feature '{feature}' not found for aggregation.")
for feature in binary_flag_features_to_agg:
    if feature in df.columns:
        aggregation_dict[f'{feature.strip()}_occurred'] = (feature, 'max')
    else:
        print(f"Warning: Binary flag feature '{feature}' not found for aggregation.")
if target_for_aggregation in df.columns:
    aggregation_dict['rain_daily'] = (target_for_aggregation, 'max')
else:
    print(f"Error: Target column '{target_for_aggregation}' not found. Exiting.")
    exit()

derived_df = df.groupby('date_only').agg(**aggregation_dict).reset_index()
derived_df['rain_daily'] = derived_df['rain_daily'].astype(int)
derived_df = derived_df.dropna(axis=1, how='all')

print(f"Derived daily dataset shape: {derived_df.shape}")
print("\nFirst 5 rows of the derived dataset with expanded features:")
print(derived_df.head())
print(f"Initial class distribution of 'rain_daily':\n{derived_df['rain_daily'].value_counts()}")

X = derived_df.drop(columns=['date_only', 'rain_daily'], errors='ignore')
y = derived_df['rain_daily']

print("\n--- Final NaN Imputation in X before splitting ---")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
            print(f"Imputed remaining missing values in '{col}' with median.")
print(f"Missing values in X after final imputation: {X.isnull().sum().sum()}")

numerical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numerical_features)],
    remainder='passthrough'
)

target_names_for_report = ['No Rain', 'Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Data split into training and testing sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Random Forest Model Training and Evaluation with Hyperparameter Tuning ---
print("\n--- Random Forest Model Training and Evaluation with Hyperparameter Tuning ---")

param_distributions_rf = {
    'classifier__n_estimators': randint(100, 700),
    'classifier__max_depth': randint(5, 25),
    'classifier__min_samples_split': randint(2, 15),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__class_weight': ['balanced', None]
}

rf_model = RandomForestClassifier(random_state=42)

pipeline_rf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

random_search_rf = RandomizedSearchCV(
    estimator=pipeline_rf,
    param_distributions=param_distributions_rf,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_rf.fit(X_train, y_train)
best_rf_model = random_search_rf.best_estimator_
print(f"Best parameters for Random Forest: {random_search_rf.best_params_}")
print(f"Best cross-validation accuracy for Random Forest: {random_search_rf.best_score_:.4f}")

y_pred_rf = best_rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=target_names_for_report, zero_division=0)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])

print(f"\nRandom Forest Test Accuracy: {accuracy_rf:.4f}")
print(f"\nRandom Forest Classification Report:\n{report_rf}")

# --- Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
cmp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=target_names_for_report)
cmp.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Random Forest Confusion Matrix for Rain Prediction (Tuned)')
plt.tight_layout()
plt.show()

# --- ROC Curve and AUC ---
y_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]  # Probability for class 'Rain' (1)
fpr, tpr, thresholds = roc_curve(y_test, y_proba_rf)
auc_score = roc_auc_score(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Rain Prediction)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nAUC Score for Random Forest: {auc_score:.4f}")

# --- Final Result Summary ---
print("\n--- Final Result Summary ---")
print(f"\nRandom Forest Model achieved a Test Accuracy: {accuracy_rf:.4f}")
print(f"AUC Score: {auc_score:.4f}")
