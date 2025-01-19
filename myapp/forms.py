from django import forms
from django.core.exceptions import ValidationError

# Custom validator for file extension
# def validate_file_extension(value):
#     allowed_extensions = ['.txt', '.csv']
#     if not any(value.name.lower().endswith(ext) for ext in allowed_extensions):
#         raise ValidationError("Only .txt and .csv files are allowed.")

class FileUploadForm(forms.Form):
    file = forms.FileField(
        label='Select a file',
        widget=forms.ClearableFileInput(attrs={'name': 'uploaded_file'}),
        # validators=[validate_file_extension],
        # help_text="Only .txt and .csv files are allowed."
    )
