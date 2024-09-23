from django.urls import path
from . import views

urlpatterns = [
    path('', views.render_upload_form, name='render_upload_form'),
    path('upload/', views.handle_file_upload, name='handle_file_upload'),
]