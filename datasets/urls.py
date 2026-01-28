from django.urls import path
from .views import dataset_upload_view, dataset_list_view

urlpatterns = [
    path('upload/', dataset_upload_view, name='dataset_upload'),
    path('list/', dataset_list_view, name='dataset_list'),
]
