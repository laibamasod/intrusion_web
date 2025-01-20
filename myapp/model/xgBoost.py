from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/IDSdataset_Draft11.csv')

# Assuming 'Class' is your label and the rest are features
X = df.drop('Class', axis=1)
y = df['Class']

y = y - 1
# Initialize XGBoost Classifier with best parameters
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                              objective='multi:softmax', num_class=6)

# Perform Cross-Validation using cross_val_score
cv_scores = cross_val_score(xgb_model, X, y, cv=5)  # 5-fold cross-validation

# Display the cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Get predictions
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)

# Save everything in a dictionary
model_data = {
    "model": xgb_model,
    "y_test": y_test,  # True labels
    "y_pred": y_pred,  # Predictions
    "y_prob": y_prob,  # Probabilities for ROC
    "cv_scores": cv_scores  # Cross-validation scores
}

# Save to joblib
joblib.dump(model_data, "finalized_xgboost.sav")
print("Model and test results saved successfully!")
