from django import forms

class FileUploadForm(forms.Form):
    file = forms.FileField(
        label='Select a file',
        widget=forms.ClearableFileInput(attrs={'class': 'form-control',}),
    )

    # Dropdown for selecting the model
    MODEL_CHOICES = [
        ("rf", "Random Forest"),
        ("xgb", "XGBoost"),
        ("lgb", "LightGBM"),
    ]

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Select Model",
        widget=forms.Select(attrs={'class': 'form-select'}),  # Bootstrap styling
        required=True
    )
