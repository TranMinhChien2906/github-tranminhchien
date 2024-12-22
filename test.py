import sqlite3

conn = sqlite3.connect("cvparserai.db")
cursor = conn.cursor()

cursor.execute( """
  SELECT * FROM users
  """)

for rec in cursor.fetchall():
    print("{},{},{},{}".format(*rec))