import psycopg2
import os
from werkzeug.security import generate_password_hash

# Database initialization script

def init_db():
    conn = None
    try:
        # Connect to PostgreSQL database
        # Replace with your PostgreSQL connection details or environment variables
        db_host = os.getenv('DB_HOST', 'localhost')
        db_name = os.getenv('DB_NAME', 'facesnap')
        db_user = os.getenv('DB_USER', 'facesnap_user')
        db_password = os.getenv('DB_PASSWORD', 'facesnap_password')

        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        
        # Create admins table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                email VARCHAR(255)
            )
        ''')
        
        # Check if admin user exists
        cursor.execute("SELECT COUNT(*) FROM admins")
        admin_count = cursor.fetchone()[0]
        
        # Create default admin user if none exists
        if admin_count == 0:
            default_username = 'admin'
            default_password = 'admin123'  # This should be changed after first login
            password_hash = generate_password_hash(default_password)
            
            cursor.execute(
                "INSERT INTO admins (username, password_hash, email) VALUES (%s, %s, %s)",
                (default_username, password_hash, 'admin@facesnap.local')
            )
            print(f"Created default admin user: {default_username} (password: {default_password})")
            print("IMPORTANT: Please change this password after first login!")
        
        # Commit changes
        conn.commit()
        
        print("Database initialized successfully!")

    except psycopg2.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_db()