"""
URL configuration for lib project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from lib import views
from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, {"name": "home"}), 
    path('getstarted', views.index, {"name": "getstarted"}),
    path('codesnippet', views.index, {"name": "codesnippet"}),
    path('pricing', views.index, {"name": "pricing"}),
    path('login', views.index, {"name": "login"}),
    path('signup', views.index, {"name": "signup"}),
    path('dashboard', views.dashboard),
    path('yourapikey', views.yourapikey),
    path('subcribe', views.subcribe),
]
urlpatterns += staticfiles_urlpatterns() 