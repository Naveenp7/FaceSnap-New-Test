import sqlite3
import os

def fix_image_paths():
    # Connect to the database
    conn = sqlite3.connect('instance/facesnap.sqlite')
    cursor = conn.cursor()
    
    # Get all image paths
    cursor.execute('SELECT id, file_path FROM images')
    images = cursor.fetchall()
    
    # Update paths to be relative to static folder
    for image_id, file_path in images:
        # If file_path doesn't start with uploads/, add uploads/1/
        if not file_path.startswith(('uploads/')):
            new_path = os.path.join('uploads', '1', os.path.basename(file_path))
        elif file_path.startswith('uploads/') and '1/' not in file_path:
            # If it starts with uploads/ but doesn't have the event ID folder, add it
            new_path = os.path.join('uploads', '1', os.path.basename(file_path))
        else:
            continue
        
        # Replace backslashes with forward slashes
        new_path = new_path.replace('\\', '/')
        print(f"Updating path for image {image_id}: {file_path} -> {new_path}")
        
        # Update the database
        cursor.execute('UPDATE images SET file_path = ? WHERE id = ?', (new_path, image_id))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Image paths have been fixed!")

if __name__ == '__main__':
    fix_image_paths()
