import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# create new table
cursor.execute(''' 
CREATE TABLE IF NOT EXISTS papers (
    email TEXT, 
    table_name TEXT, 
    doi TEXT,
    pmid TEXT,
    title TEXT,
    abstract TEXT,
    summary TEXT, 
    mesh_terms TEXT,
    scikit_keywords TEXT, 
    nltk_keywords TEXT,
    time TEXT
)
''')

cursor.close()
conn.close()
