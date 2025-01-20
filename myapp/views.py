from .forms import FileUploadForm
from .utils import load_model, generate_model_metrics, clean_report, generate_confusion_matrix, generate_roc_curve, generate_cross_val_plot
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import csv
from django.utils.encoding import smart_str
import numpy as np

# Load RandomForest model and its results once at startup
rf, y_test, y_pred, y_prob, cv_scores = load_model("finalized_randomForest.sav")

# Precompute images and store them globally for RandomForest
classes = np.unique(y_test)
cm_image = generate_confusion_matrix(y_test, y_pred)
roc_image = generate_roc_curve(y_test, y_prob, classes)
cv_image = generate_cross_val_plot(cv_scores)

# Generate metrics and clean report
accuracy, report = generate_model_metrics(y_test, y_pred)
report = clean_report(report)

def render_upload_form(request):
    form = FileUploadForm()
    return render(request, 'myapp/upload.html', {'form': form})

def handle_file_upload(request):
    model= request.POST.get('model').strip()
    uploaded_file = request.FILES.get('file')
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'txt':
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            return HttpResponse("Unsupported file type", status=400)

        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        if model == 'rf':
            # Make predictions using the random forest model (or any future model)
            predictions = rf.predict(df)  # change this to another model's prediction later if needed
        df['Class'] = predictions

        # Prepare the file for download
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=predictions.csv'

        writer = csv.writer(response, csv.excel)
        response.write(u'\ufeff'.encode('utf8'))  # Add BOM for Excel compatibility
        writer.writerow([smart_str(col) for col in df.columns])
        for row in df.values:
            writer.writerow([smart_str(cell) for cell in row])

        # Pass the confusion matrix and ROC curve to the frontend
        request.session['predicted_data'] = df.to_csv(index=False)
        request.session['uploaded_filename'] = uploaded_file.name

        return render(request, 'myapp/result.html', {
            "cm_image": cm_image,
            "roc_image": roc_image,
            "cv_image": cv_image,
            "accuracy": round(accuracy, 4),
            "report": report
        })
    
    return HttpResponse("No file uploaded", status=400)

def download_predictions(request):
    predicted_data = request.session.get('predicted_data', None)
    uploaded_filename = request.session.get('uploaded_filename', 'predictions.csv')
    if predicted_data:
        response = HttpResponse(predicted_data, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="predictions_{uploaded_filename}"'

        # Clear session data after download
        request.session.pop('predicted_data', None)
        request.session.pop('uploaded_filename', None)
        return response
    else:
        return HttpResponse("No predictions available", status=404)
