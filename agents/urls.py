from django.urls import path
from . import views

urlpatterns = [
    path('analyze/<int:dataset_id>/', views.agent2_analysis_view, name='agent2_analysis'),
]
