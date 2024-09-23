from django import forms

class FileUploadForm(forms.Form):
    file = forms.FileField(
        label='Select a file',
        help_text='Max. 42 megabytes'
    )
