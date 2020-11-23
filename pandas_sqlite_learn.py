import sqlite3
import pandas as pd

conn = sqlite3.connect('soccer/database.sqlite')

################################################################################################
cur = conn.cursor()

cur.execute('select name from sqlite_master where type = "table";')
tables = cur.fetchall()
print(tables)

cur.execute('select sql from sqlite_master where type = "table" and name = "League";')
tables = cur.fetchall()
print(tables)

cur.execute('select * from League limit 5;')
results = cur.fetchall()
print(results)

cur.close()
################################################################################################

query = 'select L.id as league_id, C.name as country, L.name as league from League as L, Country as C'
df = pd.read_sql_query(query, conn)
print(df.head())

conn.close()



