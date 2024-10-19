import sqlite3
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain_core.tools import tool

DB_PATH = 'agents/db.sqlite'

class RunSqliteQueryInput(BaseModel):
    query: str = Field(description='The query to run on the sqlite database')

class DescribeTablesInput(BaseModel):
    table_names: list = Field(description='The list of table names to describe')

@tool
def list_tables(args=None) -> str:
    '''Returns a comma separated list of the table names in the sqlite database'''
    print('\n****** list_tables Function called******\n')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = cursor.fetchall()
    return ", ".join(row[0] for row in rows if row[0] is not None)

@tool
def run_sqlite_query(query: str) -> list:
    '''Accepts a query and runs it on the sqlite database'''
    print('\n****** run_sqlite_query Function called with:', query, '******\n')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()

@tool
def describe_tables(table_names) -> str:
    '''Describe the tables in the sqlite database by accepting a list of table names. Returns the CREATE TABLE statement for each table'''
    print('\n****** describe_tables Function called with:', table_names, '******\n')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    tables = ','.join("'" + table_name + "'" for table_name in table_names)
    query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});"
    print('Query :', query)
    rows = cursor.execute(query)
    return '\n'.join(row[0] for row in rows if row[0] is not None)

@tool
def write_html_report(file_name: str, html_content: str):
    '''Write the html content to a file.'''
    print('\n****** write_html_report Function called with:', file_name, '******\n')
    with open(file_name, 'w') as f:
        f.write(html_content)

# print(run_sqlite_query('SELECT * FROM users'))

# Tool.from_function is not working properly with registering tools using the bind_tools method 
# So, using the @tool decorator to register tools
# without it the model is not able to recognize the tools

# run_query_tool = Tool.from_function(
#     name='run_sqlite_query',
#     description='Run a query on the sqlite database',
#     func=run_sqlite_query,
#     args_schema=RunSqliteQueryInput
# )

# print(list_tables())
# print(describe_tables(['products']))

# describe_tables_tool = Tool.from_function(
#     name='describe_tables',
#     description='Describe the tables in the sqlite database by accepting a list of table names',
#     func=describe_tables,
#     args_schema=DescribeTablesInput
# )

# list_tables_tool = Tool.from_function(
#     name='list_tables',
#     description='Returns a comma separated list of the table names in the sqlite database. Does not accept any arguments',
#     func=list_tables,
#     args_schema=None
# )