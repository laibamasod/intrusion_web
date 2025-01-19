import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
# Load dataset
df = pd.read_csv('data/IDSdataset_Draft11.csv')

X = df.drop('Class', axis=1)
y = df['Class']


# Initialize the Random Forest Classifier with max_depth
rf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)

scores = cross_val_score(rf, X, y, cv=5)  # 5-fold cross-validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Get predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)  # Needed for ROC curve


# Save everything in a dictionary
model_data = {
    "model": rf,
    "y_test": y_test,  # True labels
    "y_pred": y_pred,  # Predictions
    "y_prob": y_prob,  # Probabilities for ROC
    "cv_scores": scores
}

# Save to joblib
joblib.dump(model_data, "finalized_randomForest.sav")
print("Model and test results saved successfully!")