import sqlite3

conn = sqlite3.connect("cvparserai.db", check_same_thread=False)
cursor = conn.cursor()

def executeSQL(SQL):
    cursor.execute(SQL)
    conn.commit()

def insertAccount(email, password, salt):
    cursor.execute("INSERT INTO users (id, email, password, salt) VALUES (NULL,'{}','{}','{}')".format(email, password, salt))
    conn.commit()

def get(SQL):
    cursor.execute(SQL)
    return cursor.fetchall()
