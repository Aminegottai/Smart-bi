from django.urls import path
from .views import home_view, home_user_view, dashboard_view, loading_view, check_analysis_status,chatbot_view
from django.conf import settings
from django.conf.urls.static import static

app_name = "core"

urlpatterns = [
    path('', home_view, name='home'),
    path('dashboard/', home_user_view, name='dashboard'),
    path('dashboard_bi/<int:dataset_id>/', dashboard_view, name='dashboard_bi'),
    path('loading/<int:dataset_id>/', loading_view, name='loading'),
    path('check_status/<int:dataset_id>/', check_analysis_status, name='check_status'),
    path("chatbot/<int:dataset_id>/", chatbot_view, name="chatbot"),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
