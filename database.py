import pymysql

DB_CFG = dict(host="127.0.0.1", user="root", password="0000", database="pocketc", charset="utf8mb4", autocommit=False)

def get_conn():
    return pymysql.connect(**DB_CFG, cursorclass=pymysql.cursors.DictCursor)
