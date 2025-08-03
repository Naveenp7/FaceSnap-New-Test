import os
import os
import pickle
import uuid
import shutil
import tempfile
import requests
import psycopg2
import psycopg2.extras
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, session, g, abort, jsonify, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from face_engine import FaceEngine
import utils

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_change_in_production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
app.config['DATABASE'] = os.environ.get('DATABASE_URL', 'postgresql://user:password@host:port/database_name')

# Context processor to provide common variables to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Initialize face recognition engine
face_engine = FaceEngine(app.config['DATABASE'])

# Database connection handling
def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(app.config['DATABASE'])
        g.db.autocommit = True
        g.db.row_factory = psycopg2.extras.DictRow
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        
        user = db.execute('SELECT * FROM admins WHERE username = %s', (username,)).fetchone()
        
        if user is None:
            error = 'Invalid username'
        elif not check_password_hash(user['password_hash'], password):
            error = 'Invalid password'
            
        if error is None:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))
            
        flash(error, 'error')
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    # Get all events created by the current user
    events = db.execute(
        'SELECT * FROM events WHERE created_by = %s ORDER BY date DESC', 
        (session['user_id'],)
    ).fetchall()
    
    return render_template('dashboard.html', events=events)

@app.route('/events/<int:event_id>/delete', methods=['POST'])
@login_required
def delete_event(event_id):
    db = get_db()
    
    # Verify the event exists and belongs to the current user
    event = db.execute('SELECT * FROM events WHERE id = %s AND created_by = %s',
                      (event_id, session['user_id'])).fetchone()
    
    if event is None:
        abort(404)
    
    try:
        # Begin transaction
        db.execute('BEGIN')
        
        # Delete all access logs for this event
        db.execute('DELETE FROM access_logs WHERE event_id = %s', (event_id,))
        
        # Delete all face crops for images in this event
        db.execute('''
            DELETE FROM face_crops 
            WHERE image_id IN (SELECT id FROM images WHERE event_id = %s)
        ''', (event_id,))
        
        # Delete all users associated with this event
        db.execute('DELETE FROM users WHERE event_id = %s', (event_id,))
        
        # Delete all face clusters for this event
        db.execute('DELETE FROM face_clusters WHERE event_id = %s', (event_id,))
        
        # Delete all images for this event
        db.execute('DELETE FROM images WHERE event_id = %s', (event_id,))
        
        # Finally delete the event itself
        db.execute('DELETE FROM events WHERE id = %s', (event_id,))
        
        # Commit the transaction
        db.execute('COMMIT')
        
        # Delete images from Cloudinary
        from cloud_storage import delete_file
        images_to_delete = db.execute('SELECT file_path FROM images WHERE event_id = %s', (event_id,)).fetchall()
        for img in images_to_delete:
            # Extract public_id from Cloudinary URL
            # Assuming file_path is a Cloudinary URL like 'https://res.cloudinary.com/cloud_name/image/upload/v12345/public_id.jpg'
            # We need to get 'public_id' from it.
            # A more robust way would be to store public_id directly in the DB during upload.
            # For now, let's try to parse it.
            try:
                public_id_with_extension = img['file_path'].split('/')[-1]
                public_id = public_id_with_extension.rsplit('.', 1)[0]
                # Cloudinary folder structure: facesnap/events/{event_id}/public_id
                full_public_id = f"facesnap/events/{event_id}/{public_id}"
                delete_file(full_public_id)
            except Exception as e:
                app.logger.error(f"Error extracting public_id or deleting from Cloudinary: {e}")

        # Delete local physical files (if any remain or for other static assets)
        faces_dir = os.path.join('static/faces', str(event_id))
        selfies_dir = os.path.join('static/selfies', str(event_id))
        qr_file = os.path.join('static/qrcodes', f'event_{event_id}.png')
        
        # Remove directories if they exist
        for dir_path in [faces_dir, selfies_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
        # Remove QR code if it exists
        if os.path.exists(qr_file):
            os.remove(qr_file)
        
        flash('Event and all associated data have been deleted successfully', 'success')
        
    except Exception as e:
        db.execute('ROLLBACK')
        app.logger.error(f"Error deleting event: {str(e)}")
        flash('An error occurred while deleting the event', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/events/create', methods=['GET', 'POST'])
@login_required
def create_event():
    if request.method == 'POST':
        name = request.form['name']
        date = request.form['date']
        location = request.form['location']
        description = request.form.get('description', '')
        
        error = None
        
        if not name:
            error = 'Event name is required'
        elif not date:
            error = 'Event date is required'
            
        if error is None:
            db = get_db()
            db.execute(
                'INSERT INTO events (name, date, location, description, created_by) VALUES (%s, %s, %s, %s, %s)',
                (name, date, location, description, session['user_id'])
            )
            
            # Get the ID of the newly created event
            event_id = db.execute('SELECT lastval()').fetchone()[0]
            
            # Generate QR code for the event
            base_url = request.host_url.rstrip('/')
            qr_path, verify_url = utils.generate_event_qr(event_id, base_url)
            
            flash('Event created successfully', 'success')
            return redirect(url_for('event_detail', event_id=event_id))
            
        flash(error, 'error')
        
    return render_template('create_event.html')

@app.route('/events/<int:event_id>')
@login_required
def event_detail(event_id):
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    # Check if the current user created this event
    if event['created_by'] != session['user_id']:
        abort(403)
        
    # Get all images for this event
    images = db.execute(
        "SELECT i.id, REPLACE(i.file_path, '\\', '/') AS file_path, i.cluster_id, i.created_at, COUNT(DISTINCT fc.id) as face_count FROM images i "
        "LEFT JOIN face_clusters fc ON i.event_id = fc.event_id "
        "WHERE i.event_id = %s GROUP BY i.id", 
        (event_id,)
    ).fetchall()
    
    # Get all face clusters for this event
    clusters = db.execute(
        'SELECT fc.*, COUNT(i.id) as image_count, u.name as user_name '
        'FROM face_clusters fc '
        'LEFT JOIN images i ON fc.id = i.cluster_id '
        'LEFT JOIN users u ON fc.user_id = u.id '
        'WHERE fc.event_id = %s GROUP BY fc.id', 
        (event_id,)
    ).fetchall()
    
    # Generate QR code URL
    base_url = request.host_url.rstrip('/')
    qr_path, verify_url = utils.generate_event_qr(event_id, base_url)
    
    return render_template(
        'event_detail.html', 
        event=event, 
        images=images, 
        clusters=clusters,
        qr_path=qr_path,
        verify_url=verify_url
    )

@app.route('/events/<int:event_id>/upload', methods=['GET', 'POST'])
@login_required
def upload_images(event_id):
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    # Check if the current user created this event
    if event['created_by'] != session['user_id']:
        abort(403)
        
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'photos' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        files = request.files.getlist('photos')
        
        if not files or files[0].filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        # Process each uploaded file
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(event_id))
        os.makedirs(upload_dir, exist_ok=True)
        
        processed_count = 0
        face_count = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                # Upload the file to Cloudinary
                from cloud_storage import upload_file
                filename = secure_filename(file.filename)
                # Create a temporary file to save the uploaded image before sending to Cloudinary
                temp_file_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(temp_file_path)
                
                cloudinary_url, cloudinary_public_id = upload_file(temp_file_path, folder=f"facesnap/events/{event_id}")
                os.remove(temp_file_path) # Clean up temporary file

                if not cloudinary_url:
                    flash(f'Error uploading image {filename} to Cloudinary', 'error')
                    continue
                
                file_path = cloudinary_url # Use Cloudinary URL as file_path for processing
                
                # Process the image with face recognition
                try:
                    results = face_engine.process_image(file_path, event_id)
                    processed_count += 1
                    face_count += len(results)
                except Exception as e:
                    flash(f'Error processing image {filename}: {str(e)}', 'error')
        
        if processed_count > 0:
            flash(f'Successfully uploaded {processed_count} images with {face_count} faces detected', 'success')
        else:
            flash('No images were processed successfully', 'warning')
            
        return redirect(url_for('event_detail', event_id=event_id))
        
    return render_template('upload.html', event=event)

@app.route('/events/<int:event_id>/clusters')
@login_required
def view_clusters(event_id):
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    # Check if the current user created this event
    if event['created_by'] != session['user_id']:
        abort(403)
        
    # Get all face clusters for this event
    clusters = db.execute(
        'SELECT fc.*, COUNT(i.id) as image_count, u.name as user_name '
        'FROM face_clusters fc '
        'LEFT JOIN images i ON fc.id = i.cluster_id '
        'LEFT JOIN users u ON fc.user_id = u.id '
        'WHERE fc.event_id = %s GROUP BY fc.id', 
        (event_id,)
    ).fetchall()
    
    return render_template('clusters.html', event=event, clusters=clusters)

@app.route('/events/<int:event_id>/clusters/<int:cluster_id>')
@login_required
def cluster_detail(event_id, cluster_id):
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    # Check if the current user created this event
    if event['created_by'] != session['user_id']:
        abort(403)
        
    # Get the cluster
    cluster = db.execute(
        'SELECT fc.*, u.name as user_name, u.email as user_email '
        'FROM face_clusters fc '
        'LEFT JOIN users u ON fc.user_id = u.id '
        'WHERE fc.id = %s AND fc.event_id = %s', 
        (cluster_id, event_id)
    ).fetchone()
    
    if cluster is None:
        abort(404)
        
    # Get all images for this cluster
    images = db.execute(
        "SELECT id, REPLACE(REPLACE(file_path, 'static/', ''), '\\', '/') as file_path, cluster_id, created_at FROM images WHERE cluster_id = %s AND event_id = %s", 
        (cluster_id, event_id)
    ).fetchall()

    # Get a few sample faces for the cluster (e.g., first 6 face crops)
    sample_faces = db.execute(
        "SELECT id, REPLACE(REPLACE(file_path, 'static/', ''), '\\', '/') as file_path, cluster_id, image_id, created_at FROM face_crops WHERE cluster_id = %s LIMIT 6",
        (cluster_id,)
    ).fetchall()
    
    # Convert sample faces into dictionary format
    sample_faces = [{'file_path': face['file_path']} for face in sample_faces]

    # Generate QR code
    base_url = request.host_url.rstrip('/')
    qr_code_path = utils.generate_cluster_qr(event_id, cluster_id, base_url)

    return render_template('cluster_detail.html', event=event, cluster=cluster, images=images, sample_faces=sample_faces)

@app.route('/event/verify')
def verify_page():
    event_id = request.args.get('id')
    
    if not event_id:
        abort(404)
        
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    return render_template('verify.html', event=event)

@app.route('/event/verify/<int:event_id>', methods=['POST'])
def verify_user(event_id):
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    # Get form data
    name = request.form['name']
    email = request.form.get('email', '')
    phone = request.form.get('phone', '')
    
    # Check if selfie data was provided
    selfie_path = None
    
    # Check if a selfie was uploaded as a file
    if 'selfie' in request.files and request.files['selfie'].filename != '':
        selfie = request.files['selfie']
        
        if not allowed_file(selfie.filename):
            flash('Invalid file type', 'error')
            return redirect(url_for('verify_page', id=event_id))
            
        # Save the selfie temporarily
        selfie_dir = os.path.join('static', 'selfies', str(event_id))
        os.makedirs(selfie_dir, exist_ok=True)
        
        # Get the full path where the file is saved
        full_selfie_path = utils.save_uploaded_file(selfie, selfie_dir)
        
        # Store the relative path for database storage
        db_selfie_path = os.path.relpath(full_selfie_path, 'static')    
    # Check if selfie was provided as base64 data
    elif 'selfie_data' in request.form and request.form['selfie_data']:
        import base64
        import re
        
        # Get the base64 data
        selfie_data = request.form['selfie_data']
        
        # Extract the actual base64 content
        if selfie_data.startswith('data:image'):
            # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/...
            selfie_data = re.sub('^data:image/[^;]+;base64,', '', selfie_data)
        
        # Decode the base64 data
        try:
            selfie_bytes = base64.b64decode(selfie_data)
            
            # Save the selfie temporarily
            selfie_dir = os.path.join('static', 'selfies', str(event_id))
            os.makedirs(selfie_dir, exist_ok=True)
            
            # Generate a unique filename
            selfie_filename = f"{uuid.uuid4()}.jpg"
            full_selfie_path = os.path.join(selfie_dir, selfie_filename)
            
            # Write the image to a file
            with open(full_selfie_path, 'wb') as f:
                f.write(selfie_bytes)
                
            # Store the relative path for database storage
            db_selfie_path = os.path.relpath(full_selfie_path, 'static')
                
        except Exception as e:
            app.logger.error(f"Error processing selfie data: {str(e)}")
            flash('Error processing selfie data', 'error')
            return redirect(url_for('verify_page', id=event_id))
    
    else:
        flash('No selfie provided', 'error')
        return redirect(url_for('verify_page', id=event_id))
    
    # Verify the user's face
    verification_result = face_engine.verify_user(full_selfie_path, event_id)
    
    if verification_result['success']:
        # User verified successfully
        cluster_id = verification_result['cluster_id']
        
        # Save user information
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO users (name, email, phone, cluster_id, event_id, selfie_path) VALUES (%s, %s, %s, %s, %s, %s)',
            (name, email, phone, cluster_id, event_id, db_selfie_path)
        )
        user_id = cursor.fetchone()[0]
        
        # Update the face cluster with the user ID
        db.execute(
            'UPDATE face_clusters SET user_id = %s WHERE id = %s',
            (user_id, cluster_id)
        )
        
        # Log the access
        utils.log_access(user_id, event_id, cluster_id, request.remote_addr, db)
        
        # Redirect to the gallery
        return redirect(url_for('gallery', event_id=event_id, cluster_id=cluster_id))
    else:
        # Verification failed
        flash(verification_result['message'], 'error')
        return redirect(url_for('verify_page', id=event_id))

@app.route('/gallery/<int:event_id>/<int:cluster_id>')
def gallery(event_id, cluster_id):
    db = get_db()
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    
    if event is None:
        abort(404)
        
    # Get the cluster
    cluster = db.execute(
        'SELECT * FROM face_clusters WHERE id = %s AND event_id = %s', 
        (cluster_id, event_id)
    ).fetchone()
    
    if cluster is None:
        abort(404)
    
    # Get the user associated with this cluster
    user = db.execute(
        'SELECT * FROM users WHERE cluster_id = %s AND event_id = %s ORDER BY id DESC LIMIT 1', 
        (cluster_id, event_id)
    ).fetchone()
    
    if user is None:
        # If no user is found, create a placeholder to avoid template errors
        user = {'name': 'Guest', 'selfie_path': '../img/default_avatar.svg'}
        
    # Get all images for this cluster
    images = db.execute(
        "SELECT id, REPLACE(REPLACE(file_path, 'static/', ''), '\\', '/') as file_path, cluster_id, created_at FROM images WHERE cluster_id = %s AND event_id = %s", 
        (cluster_id, event_id)
    ).fetchall()
    
    return render_template('gallery.html', event=event, cluster=cluster, images=images, user=user)

@app.route('/download/<int:image_id>')
def download_image(image_id):
    db = get_db()
    image = db.execute('SELECT * FROM images WHERE id = %s', (image_id,)).fetchone()
    
    if image is None:
        app.logger.error(f"Image {image_id} not found in database")
        abort(404)
    
    # The file_path now contains the Cloudinary URL
    cloudinary_url = image['file_path']
    
    # Get original filename from the URL for download_name
    filename = os.path.basename(cloudinary_url.split('?')[0]) # Remove query parameters if any
    
    # Redirect to Cloudinary URL for download
    return redirect(cloudinary_url)


@app.route('/download-all/<int:event_id>/<int:cluster_id>')
def download_all(event_id, cluster_id):
    import zipfile
    import tempfile
    import shutil
    
    db = get_db()
    
    # Verify event and cluster exist
    event = db.execute('SELECT * FROM events WHERE id = %s', (event_id,)).fetchone()
    cluster = db.execute('SELECT * FROM face_clusters WHERE id = %s AND event_id = %s', 
                        (cluster_id, event_id)).fetchone()
    
    if event is None or cluster is None:
        abort(404)
    
    # Get all images for this cluster
    images = db.execute(
        'SELECT * FROM images WHERE cluster_id = %s AND event_id = %s', 
        (cluster_id, event_id)
    ).fetchall()
    
    if not images:
        flash('No images found in this cluster', 'warning')
        return redirect(url_for('gallery', event_id=event_id, cluster_id=cluster_id))
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a zip file
        zip_filename = f"event_{event_id}_cluster_{cluster_id}_photos.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        import requests
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for image in images:
                cloudinary_url = image['file_path']
                try:
                    response = requests.get(cloudinary_url, stream=True)
                    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                    
                    # Get original filename from the URL
                    filename = os.path.basename(cloudinary_url.split('?')[0])
                    
                    # Write the image to a temporary file
                    temp_image_path = os.path.join(temp_dir, filename)
                    with open(temp_image_path, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                    
                    # Add watermark to the image
                    watermarked_path = utils.add_watermark(temp_image_path)
                    
                    # Add to zip file with a unique name
                    zipf.write(watermarked_path, arcname=filename)
                    os.remove(temp_image_path) # Clean up temporary image file
                    os.remove(watermarked_path) # Clean up watermarked image file

                except requests.exceptions.RequestException as e:
                    app.logger.error(f"Error downloading image from Cloudinary {cloudinary_url}: {e}")
                except Exception as e:
                    app.logger.error(f"Error processing image {cloudinary_url} for zip: {e}")
        
        # Send the zip file
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
    
    except Exception as e:
        app.logger.error(f"Error creating zip file: {str(e)}")
        flash('An error occurred while preparing your download', 'error')
        return redirect(url_for('gallery', event_id=event_id, cluster_id=cluster_id))
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# Helper functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Create required directories and initialize db
def init_app():
    # Create additional required directories
    os.makedirs('static/selfies', exist_ok=True)
    os.makedirs('static/qrcodes', exist_ok=True)

    os.makedirs('static/faces', exist_ok=True)
    
    # Check if database exists, if not initialize it
    try:
        conn = psycopg2.connect(app.config['DATABASE'])
        conn.close()
    except psycopg2.OperationalError:
        from init_db import init_db
        init_db()

# Initialize the app when imported
init_app()

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
