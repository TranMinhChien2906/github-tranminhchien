from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from .forms import Account
from .forms import CVFile

def index(request, name):
    if request.method == "POST":
        form = CVFile(request.POST, request.FILES)
        if form.is_valid():
            CVFile.upload(request)
            return HttpResponseRedirect("/")
    else:
        form = CVFile()
    
    context = {
        "form": form,
        "name": name
    }
    return render(request,'index.html', context)
def payment(request, name):
    return render(request, 'payment.html', {"name":name})
def signup(request, name):
    if request.method == "POST":
        form  = Account(request.POST)

        if form.is_valid():
            Account.signUp(request, form)

            return HttpResponseRedirect("/")
    else:
        form = Account()
    
    context = {
        "form": form,
        "name": "signup"
    }

    return render(request, "index.html", context)
def login(request, name):
    if request.method == "POST":
        form  = Account(request.POST)

        if form.is_valid():
            Account.login(request, form)

            return HttpResponseRedirect("/")
    else:
        form = Account()
    
    context = {
        "form": form,
        "name": "login"
    }

    return render(request, "index.html", context)
def logout(request, name):
    request.session["loginStatus"] = False
    return render(request, "index.html", {"name":"logout"})

def dashboard(request):
    return HttpResponse("dashboard")
def yourapikey(request):
    return HttpResponse("yourapikey")
def subcribe(request):
    return HttpResponse("subcribe")