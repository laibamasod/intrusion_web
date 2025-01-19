import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score

# Load model & stored test results once at startup
def load_model():
    model_data = joblib.load("finalized_randomForest.sav")
    return model_data["model"], model_data["y_test"], model_data["y_pred"], model_data["y_prob"], model_data["cv_scores"]

# Clean the classification report
def clean_report(report):
    for label, values in report.items():
        if isinstance(values, dict):
            if 'f1-score' in values:
                values['f1_score'] = values.pop('f1-score')
    return report

# Generate model metrics
def generate_model_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report.get('accuracy', acc)
    return accuracy, report

# Generate cross-validation plot
def generate_cross_val_plot(cv_scores):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='--', color='b')
    plt.title('Cross-Validation Scores')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.grid(True)
    return save_plot_to_base64()

# Generate confusion matrix plot
def generate_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return save_plot_to_base64()

# Generate ROC curve plot
def generate_roc_curve(y_test, y_prob, classes):
    plt.figure(figsize=(6, 4))
    for i in classes:
        class_index = i - 1
        if np.any(y_test == i):
            fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, class_index])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc='lower right')
    return save_plot_to_base64()

# Save plot to base64
def save_plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    return image_base64
