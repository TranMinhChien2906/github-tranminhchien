
import hashlib
import random

alphabets = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXZ"

def getSalt():
    return ''.join(random.choice(alphabets) for i in range(16))

def hash(password, salt):
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def verifyHash(testPass, password, salt):
    newPass = hashlib.sha256((testPass + salt).encode('utf-8')).hexdigest()
    return newPass == password