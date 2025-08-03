import sqlite3

def upgrade_database(db_path='instance/facesnap.sqlite', upgrade_script='upgrade_schema.sql'):
    """Upgrades the database schema by executing a SQL script."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        with open(upgrade_script, 'r') as f:
            sql_script = f.read()
        cursor.executescript(sql_script)
        conn.commit()
        conn.close()
        print("Database upgrade successful.")
    except sqlite3.Error as e:
        print(f"Database error during upgrade: {e}")
    except FileNotFoundError:
        print(f"Upgrade script '{upgrade_script}' not found.")

if __name__ == '__main__':
    upgrade_database()
