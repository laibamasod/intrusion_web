from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import FileUploadForm

def render_upload_form(request):
    print('called render')
    form = FileUploadForm()
    return render(request, 'myapp/upload.html', {'form': form})

def handle_file_upload(request):
    print('called post')
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the file here (when you integrate with the backend)
            # For now, we'll just return a success message
            return HttpResponse("File uploaded successfully!")
    return redirect('render_upload_form')
