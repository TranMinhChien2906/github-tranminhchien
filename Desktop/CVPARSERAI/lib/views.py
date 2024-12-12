from django.http import HttpResponse
from django.shortcuts import render
def index(request, name):
    return render(request,'index.html', {"name":name})
def dashboard(request):
    return HttpResponse("dashboard")
def yourapikey(request):
    return HttpResponse("yourapikey")
def subcribe(request):
    return HttpResponse("subcribe")