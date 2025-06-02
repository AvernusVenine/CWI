import sqlite3

desc_dict = {}

sql_conn = sqlite3.connect('compiled_data/hennepin_norm.db')
sql_cursor = sql_conn.cursor()


