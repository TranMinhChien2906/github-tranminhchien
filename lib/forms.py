import requests
import json
from django import forms
from modules import account
from modules import db
from modules.pdfreader import extract_pdf_text
import os

class CVFile(forms.Form):
    cv = forms.FileField(
        widget=forms.ClearableFileInput(
            attrs={
                "class": "btn-secondary",
                "onchange": "javascript:this.form.submit();"
                }
        ),
        label=''
    )

    def parse(cv):
        #gay = "C:\\Users\\ACER\\Desktop\\CVPARSERAI\\lib\\static\\cv\\" + cv
        #url = "http://127.0.0.1:8000/static/cv/" + cv.name
        #data = {"url": url}
        ##r = requests.post(url="http://127.0.0.1:9000/test?url=" + url, data=data)
        #print(r)
        #print(r.text)
        #return r.text
        print(extract_pdf_text(cv))
        


    def upload(request):
        uploadedFile = request.FILES["cv"].name
        fileName = os.path.join("lib/static/cv/", uploadedFile)
        fout = open(fileName, "wb+")
        for chunk in request.FILES["cv"].chunks():
            fout.write(chunk)
        fout.close()
        return fileName

class Account(forms.Form):
    email = forms.EmailField(label="Email", max_length=100)
    password = forms.CharField(label="Password", widget=forms.PasswordInput())
    
    def signUp(request, form):
        email = form.cleaned_data["email"]
        password = form.cleaned_data["password"]

        salt = account.getSalt()

        db.insertAccount(email, account.hash(password, salt), salt)

        request.session["loginStatus"] = True

    def login(request, form):
        email = form.cleaned_data["email"]
        password = form.cleaned_data["password"]

        checkAcc = db.get("SELECT password, salt FROM users where email == '{}'".format(email))

        if len(checkAcc) == 0:
            return False
            
        dbPass = checkAcc[0][0]
        salt = checkAcc[0][1]

        if account.verifyHash(password, dbPass, salt) == True:
            request.session["loginStatus"] = True

        