import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "company_data.db")

def init_db():
    print(f"Initializing dummy database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            hire_date TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS revenue (
            year INTEGER PRIMARY KEY,
            q1 INTEGER,
            q2 INTEGER,
            q3 INTEGER,
            q4 INTEGER,
            total_revenue INTEGER
        )
    """)

    # Check if empty, then insert data
    cursor.execute("SELECT COUNT(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        employees = [
            (1, 'Alice Smith', 'Engineering', 120000, '2021-03-15'),
            (2, 'Bob Johnson', 'Sales', 95000, '2022-01-10'),
            (3, 'Charlie Brown', 'Engineering', 110000, '2021-08-22'),
            (4, 'Diana Prince', 'HR', 85000, '2020-11-01'),
            (5, 'Evan Wright', 'Marketing', 90000, '2023-05-12')
        ]
        cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)
        
        revenues = [
            (2021, 500000, 600000, 550000, 700000, 2350000),
            (2022, 650000, 700000, 800000, 950000, 3100000),
            (2023, 1000000, 1100000, 1250000, 1500000, 4850000)
        ]
        cursor.executemany("INSERT INTO revenue VALUES (?, ?, ?, ?, ?, ?)", revenues)
        
        conn.commit()
        print("Inserted mock data into database.")
    else:
        print("Database already contains data.")

    conn.close()

if __name__ == "__main__":
    init_db()
