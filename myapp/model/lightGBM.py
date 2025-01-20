import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

# Load dataset
df = pd.read_csv('data/IDSdataset_Draft11.csv')

X = df.drop('Class', axis=1)
y = df['Class']

# Define the model with the best parameters
best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'num_leaves': 10}
lgb_model = lgb.LGBMClassifier(**best_params, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(lgb_model, X, y, cv=5)  # 5-fold cross-validation

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
lgb_model.fit(X_train, y_train)

# Get predictions
y_pred = lgb_model.predict(X_test)
y_prob = lgb_model.predict_proba(X_test)

# Save everything in a dictionary
model_data = {
    "model": lgb_model,
    "y_test": y_test,  # True labels
    "y_pred": y_pred,  # Predictions
    "y_prob": y_prob,  # Probabilities for ROC
    "cv_scores": cv_scores
}

# Save to joblib
joblib.dump(model_data, "finalized_lightGBM.sav")
print("Model and test results saved successfully!")
