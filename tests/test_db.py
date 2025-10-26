import sqlite3
import os

# Test database connections
db_files = ["db/mimicsql.db", "db/mimicsql_full.db", "db/umls.db"]

for db_file in db_files:
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = cursor.fetchall()
            print(f"{db_file}: {len(tables)} tables")
            for (table_name,) in tables[:3]:  # Show first 3 tables
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                print(f"  - {table_name}: {count} rows")
            conn.close()
        except Exception as e:
            print(f"Error with {db_file}: {e}")
    else:
        print(f"{db_file}: File not found")
