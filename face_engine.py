import os
import face_recognition
import numpy as np
import cv2
from PIL import Image
import sqlite3
import pickle
import uuid
from datetime import datetime
import logging
import dlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEngine:
    def __init__(self, db_path='instance/facesnap.sqlite', model='hog'):
        self.db_path = db_path
        self.face_similarity_threshold = 0.55  # Lowered threshold for better matching
        self.model = model
        self.min_face_size = 40  # Reduced minimum face size for better detection
        self.face_padding = 0.2   # 20% padding around detected faces
        self.min_detection_confidence = 0.8  # Slightly reduced for better detection
        self.max_image_size = 1600  # Maximum image dimension for processing
        self.min_image_size = 200  # Minimum image dimension
        
        # Configure folder paths
        self.upload_dir = 'static/uploads'
        self.faces_dir = 'static/faces'
        self.selfies_dir = 'static/selfies'
        
        # Ensure required directories exist
        for directory in [self.upload_dir, self.faces_dir, self.selfies_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Set up logging
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the FaceEngine"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger('FaceEngine')
        
    def _get_db_connection(self):
        """Get a connection to the SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise
            
    def ensure_event_directories(self, event_id):
        """Ensure all required directories exist for an event"""
        event_dirs = [
            os.path.join(self.upload_dir, str(event_id)),
            os.path.join(self.faces_dir, str(event_id)),
            os.path.join(self.selfies_dir, str(event_id))
        ]
        for directory in event_dirs:
            os.makedirs(directory, exist_ok=True)
            
    def _normalize_path(self, path):
        """Normalize file path for database storage"""
        return os.path.relpath(path, 'static').replace('\\', '/')

    def detect_faces(self, image_path):
        """Detect faces in an image and return their locations and encodings."""
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                return None, [], []

            # Load image using PIL first to check format and do initial processing
            try:
                pil_image = Image.open(image_path)
                
                # Convert to RGB if needed
                if pil_image.mode not in ('RGB', 'L'):
                    pil_image = pil_image.convert('RGB')
                
                # Check image dimensions
                width, height = pil_image.size
                if width < self.min_image_size or height < self.min_image_size:
                    self.logger.error(f"Image too small: {width}x{height}, minimum size is {self.min_image_size}x{self.min_image_size}")
                    return None, [], []
                
                # Resize if image is too large (preserving aspect ratio)
                if width > self.max_image_size or height > self.max_image_size:
                    scale = min(self.max_image_size/width, self.max_image_size/height)
                    new_size = (int(width * scale), int(height * scale))
                    pil_image = pil_image.resize(new_size, Image.LANCZOS)
                    self.logger.info(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
                
                # Convert to numpy array for face_recognition
                image = np.array(pil_image)
                
            except Exception as e:
                self.logger.error(f"Error processing image with PIL: {e}")
                return None, [], []

            # Try face detection with different approaches
            face_locations = []
            attempts = [
                (lambda img: face_recognition.face_locations(img, model=self.model), 'default'),
                (lambda img: face_recognition.face_locations(img, model=self.model, number_of_times_to_upsample=2), 'upsampled'),
                (lambda img: face_recognition.face_locations(cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)), model=self.model), 'equalized')
            ]

            for detect_func, method in attempts:
                if not face_locations:  # Only try next method if no faces found
                    try:
                        face_locations = detect_func(image)
                        if face_locations:
                            self.logger.info(f"Found faces using {method} method")
                            break
                    except Exception as e:
                        self.logger.warning(f"Face detection failed with {method} method: {e}")
                        continue

            self.logger.info(f"Found {len(face_locations)} faces in {image_path}")

            # Filter out faces that are too small
            if face_locations:
                valid_faces = []
                for loc in face_locations:
                    top, right, bottom, left = loc
                    face_height = bottom - top
                    face_width = right - left
                    if face_height >= self.min_face_size and face_width >= self.min_face_size:
                        valid_faces.append(loc)
                face_locations = valid_faces
                self.logger.info(f"{len(face_locations)} faces meet minimum size requirement")

            # Get face encodings for detected faces with error handling for each face
            face_encodings = []
            if face_locations:
                for i, face_location in enumerate(face_locations):
                    try:
                        encodings = face_recognition.face_encodings(image, [face_location])
                        if encodings:
                            face_encodings.append(encodings[0])
                        else:
                            self.logger.warning(f"Could not compute encoding for face {i+1}")
                    except Exception as e:
                        self.logger.error(f"Error encoding face {i+1}: {e}")

            return image, face_locations, face_encodings

        except Exception as e:
            self.logger.error(f"Error detecting faces in {image_path}: {e}")
            return None, [], []

    def find_or_create_cluster(self, event_id, face_encoding):
        """Find an existing face cluster or create a new one based on face similarity."""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, average_encoding, face_count FROM face_clusters WHERE event_id = ? ORDER BY face_count DESC", (event_id,))
                clusters = cursor.fetchall()

                # Score each cluster and track all distances for analysis
                cluster_scores = []
                for cluster in clusters:
                    if not cluster['average_encoding']:
                        continue
                        
                    try:
                        average_encoding = pickle.loads(cluster['average_encoding'])
                        distance = face_recognition.face_distance([average_encoding], face_encoding)[0]
                        score = {
                            'cluster_id': cluster['id'],
                            'distance': distance,
                            'face_count': cluster['face_count'],
                            'average_encoding': cluster['average_encoding']
                        }
                        cluster_scores.append(score)
                    except Exception as e:
                        self.logger.error(f"Error processing cluster {cluster['id']}: {e}")
                        continue

                # Sort clusters by distance
                cluster_scores.sort(key=lambda x: x['distance'])

                # If we have matches within threshold
                if cluster_scores and cluster_scores[0]['distance'] < self.face_similarity_threshold:
                    best_match = cluster_scores[0]
                    
                    # Check if there's a close second match that might indicate ambiguity
                    if len(cluster_scores) > 1:
                        second_best = cluster_scores[1]
                        distance_diff = second_best['distance'] - best_match['distance']
                        
                        # If the difference is very small and both are within threshold,
                        # prefer the cluster with more faces
                        if (distance_diff < 0.1 and 
                            second_best['distance'] < self.face_similarity_threshold and 
                            second_best['face_count'] > best_match['face_count'] * 1.5):
                            best_match = second_best
                    
                    # Update the chosen cluster
                    self._update_cluster_average(
                        best_match['cluster_id'], 
                        face_encoding,
                        best_match['average_encoding'],
                        best_match['face_count']
                    )
                    return best_match['cluster_id']

                # No matching cluster found, create a new one
                return self._create_new_cluster(event_id, face_encoding)

            except Exception as e:
                self.logger.error(f"Error in find_or_create_cluster: {e}")
                raise
            finally:
                conn.close()

        except sqlite3.Error as e:
            self.logger.error(f"Database error in find_or_create_cluster: {e}")
            return None

    def _create_new_cluster(self, event_id, face_encoding):
        """Creates a new cluster and initializes it with the first face encoding."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            face_encoding_binary = pickle.dumps(face_encoding)
            cursor.execute(
                "INSERT INTO face_clusters (event_id, created_at, average_encoding, face_count) VALUES (?, ?, ?, ?)",
                (event_id, datetime.now().isoformat(), face_encoding_binary, 1)
            )
            new_cluster_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return new_cluster_id
        except sqlite3.Error as e:
            logger.error(f"Database error while creating a new cluster: {e}")
            return None

    def _update_cluster_average(self, cluster_id, new_encoding, existing_average_blob, face_count):
        """Incrementally updates the average encoding for a cluster."""
        try:
            existing_average = pickle.loads(existing_average_blob)
            new_average = ((existing_average * face_count) + new_encoding) / (face_count + 1)
            new_average_binary = pickle.dumps(new_average)

            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE face_clusters SET average_encoding = ?, face_count = ? WHERE id = ?",
                (new_average_binary, face_count + 1, cluster_id)
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database error while updating cluster average: {e}")

    def save_face_crop(self, image, face_location, event_id, cluster_id):
        """Crop a face from an image and save it to the appropriate directory"""
        try:
            # Create the cluster directory
            cluster_dir = os.path.join('static', 'faces', str(event_id), f'cluster_{cluster_id}')
            os.makedirs(cluster_dir, exist_ok=True)

            # Extract face coordinates
            top, right, bottom, left = face_location
            
            # Calculate margins (20% of face size)
            height = bottom - top
            width = right - left
            margin_h = int(height * 0.2)
            margin_w = int(width * 0.2)

            # Add margins while keeping within image bounds
            img_height, img_width = image.shape[:2]
            crop_top = max(0, top - margin_h)
            crop_bottom = min(img_height, bottom + margin_h)
            crop_left = max(0, left - margin_w)
            crop_right = min(img_width, right + margin_w)

            # Extract the face region
            face_image = image[crop_top:crop_bottom, crop_left:crop_right]

            # Ensure we have a valid crop
            if face_image.size == 0:
                logger.error("Face crop resulted in empty image")
                return None

            # Convert color space if needed
            if len(face_image.shape) == 3:  # Color image
                if face_image.shape[2] == 3:  # BGR to RGB
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:  # Grayscale image
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(face_image)

            # Generate unique filename and save
            filename = f"{uuid.uuid4()}.jpg"
            file_path = os.path.join(cluster_dir, filename)
            
            # Save with error handling
            try:
                pil_image.save(file_path, quality=95)
                logger.info(f"Successfully saved face crop to {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Error saving face crop: {e}")
                return None

        except Exception as e:
            logger.error(f"Error in save_face_crop: {e}")
            return None

    def process_image(self, image_path, event_id):
        """Process an uploaded image, detect faces, and assign to clusters"""
        logger.info(f"Processing image: {image_path} for event: {event_id}")
        results = []
        
        try:
            # Ensure the image path exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return results
                
            # First try to load the image to verify it's valid
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return results
                
            # Convert to RGB for face_recognition library
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(image_rgb, model=self.model)
            logger.info(f"Found {len(face_locations)} faces in image")
            
            if not face_locations:
                logger.warning(f"No faces detected in image: {image_path}")
                return results
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
            
            # Process each detected face
            for face_location, face_encoding in zip(face_locations, face_encodings):
                try:
                    # Find or create a cluster for this face
                    cluster_id = self.find_or_create_cluster(event_id, face_encoding)
                    if cluster_id is None:
                        logger.warning("Failed to create or find cluster for face")
                        continue
                        
                    # Save the face crop
                    face_path = self.save_face_crop(image, face_location, event_id, cluster_id)
                    
                    # Save to database
                    try:
                        conn = self._get_db_connection()
                        cursor = conn.cursor()
                        
                        # Store the encodings
                        face_encoding_binary = pickle.dumps(face_encoding)
                        
                        # Get relative paths for database storage
                        db_image_path = os.path.relpath(image_path, 'static').replace('\\', '/')
                        db_face_path = os.path.relpath(face_path, 'static').replace('\\', '/')
                        
                        # Insert image record
                        cursor.execute(
                            "INSERT INTO images (file_path, cluster_id, event_id, created_at) VALUES (?, ?, ?, ?)",
                            (db_image_path, cluster_id, event_id, datetime.now().isoformat())
                        )
                        image_id = cursor.lastrowid
                        
                        # Insert face crop record
                        cursor.execute(
                            "INSERT INTO face_crops (file_path, face_encoding, cluster_id, image_id, created_at) VALUES (?, ?, ?, ?, ?)",
                            (db_face_path, face_encoding_binary, cluster_id, image_id, datetime.now().isoformat())
                        )
                        
                        conn.commit()
                        
                        results.append({
                            'face_location': face_location,
                            'cluster_id': cluster_id,
                            'face_path': face_path
                        })
                        
                        logger.info(f"Successfully processed face and saved to cluster {cluster_id}")
                        
                    except sqlite3.Error as e:
                        logger.error(f"Database error while saving face: {e}")
                        continue
                    finally:
                        if 'conn' in locals():
                            conn.close()
                            
                except Exception as e:
                    logger.error(f"Error processing individual face: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            
        return results


        return results

    def verify_user(self, selfie_path, event_id):
        """Verify a user by matching their selfie with existing face clusters"""
        try:
            image, face_locations, face_encodings = self.detect_faces(selfie_path)

            if not face_locations:
                return {'success': False, 'message': 'No face detected in selfie'}
            if len(face_locations) > 1:
                return {'success': False, 'message': 'Multiple faces detected in selfie. Please submit a selfie with only your face.'}

            selfie_encoding = face_encodings[0]
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, average_encoding FROM face_clusters WHERE event_id = ?", (event_id,))
            clusters = cursor.fetchall()
            conn.close()

            best_match_cluster_id = None
            best_match_distance = float('inf')

            for cluster in clusters:
                if cluster['average_encoding']:
                    average_encoding = pickle.loads(cluster['average_encoding'])
                    distance = face_recognition.face_distance([average_encoding], selfie_encoding)[0]
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_cluster_id = cluster['id']

            if best_match_cluster_id is not None and best_match_distance < self.face_similarity_threshold:
                return {
                    'success': True,
                    'cluster_id': best_match_cluster_id,
                    'confidence': 1.0 - best_match_distance
                }
            else:
                return {'success': False, 'message': 'No matching face found in our database'}

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {'success': False, 'message': f'An unexpected error occurred during verification.'}