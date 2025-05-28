import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Load full CSV
df = pd.read_csv(r"C:\Users\Neksi\Desktop\F1-winner-model\outputs\f1_ready_for_model_cleaned.csv", low_memory=False)

# Clean and enforce numeric types
columns_to_use = [
    'grid', 'position_gain',
    'avg_grid_last_5', 'avg_position_last_5', 'avg_points_last_5', 'won_race'
]

df = df[columns_to_use]

# Convert all to numeric (coerce strings to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows where target is missing or invalid (just the target)
df = df.dropna(subset=['won_race'])

# Fill any remaining feature NaNs with mean
features = [col for col in df.columns if col != 'won_race']
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

y = df['won_race']

# Print final shape before split
print(f"✅ Total rows used: {len(X)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Logistic Regression
print("\nTraining Logistic Regression...")
log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_prob_log))

# XGBoost
print("\nTraining XGBoost...")
neg, pos = y_train.value_counts()
scale_pos = neg / pos

xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))

import joblib
joblib.dump(xgb_model, "xgb_model.pkl")
print("✅ XGBoost model saved as xgb_model.pkl")
