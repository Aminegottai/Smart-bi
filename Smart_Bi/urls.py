from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('datasets/', include('datasets.urls')),
    path('acountes/', include('acountes.urls')),
    path('', include('core.urls')),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)