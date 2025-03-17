from django.urls import path
from .views import get_ecs_mapping

urlpatterns = [
    path('mappings/ecs/', get_ecs_mapping),
]