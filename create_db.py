"""=================================================================================================
Script is used to create a database file called database.db. Inside there is a table called papers 
which stores the user's email, table name (for callback), paper doi, pmid, title, abstract, summary,
mesh terms, scikit keywords, nltk keywords, and the time added. 

Rose Wilfong         05/05/2023
================================================================================================="""
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
