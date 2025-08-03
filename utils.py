import os
import uuid
import qrcode
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

def save_uploaded_file(file, directory):
    """Save an uploaded file to the specified directory with a unique filename"""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate a unique filename while preserving the original extension
    original_filename = file.filename
    extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{extension}"
    
    # Save the file
    file_path = os.path.join(directory, unique_filename)
    file.save(file_path)
    
    return file_path

def generate_qr_code(url, size=200):
    """Generate a QR code for the given URL"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Create an image from the QR Code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Resize the image
    img = img.resize((size, size))
    
    # Convert to bytes for display in HTML
    img_bytes = np.array(img)
    
    return img_bytes

def generate_event_qr(event_id, base_url):
    """Generate a QR code for an event verification page"""
    # Create directory if it doesn't exist
    qr_dir = os.path.join('static', 'qrcodes')
    os.makedirs(qr_dir, exist_ok=True)
    
    # Generate the verification URL
    verify_url = f"{base_url}/event/verify?id={event_id}"
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(verify_url)
    qr.make(fit=True)

    # Create an image from the QR Code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save the QR code image
    qr_path = os.path.join(qr_dir, f"event_{event_id}.png")
    img.save(qr_path)
    
    return qr_path, verify_url

def generate_cluster_qr(event_id, cluster_id, base_url):
    """Generate a QR code for a cluster verification page"""
    # Create directory if it doesn't exist
    qr_dir = os.path.join('static', 'qrcodes')
    os.makedirs(qr_dir, exist_ok=True)
    
    # Generate the verification URL
    verify_url = f"{base_url}/event/verify/{event_id}?cluster={cluster_id}"
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(verify_url)
    qr.make(fit=True)

    # Create an image from the QR Code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save the QR code image
    qr_filename = f"cluster_{event_id}_{cluster_id}.png"
    qr_path = os.path.join('qrcodes', qr_filename)
    full_path = os.path.join('static', qr_path)
    img.save(full_path)
    
    return qr_path

def format_datetime(date_string):
    """Format a datetime string for display"""
    try:
        dt = datetime.fromisoformat(date_string)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return date_string

def add_watermark(image_path, text="FaceSnap by ALLIED"):
    """Add a watermark to an image"""
    # Load the image
    img = cv2.imread(image_path)
    
    # Get dimensions
    height, width = img.shape[:2]
    
    # Create a semi-transparent overlay
    overlay = img.copy()
    
    # Add text watermark
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = width / 1000  # Scale font based on image width
    thickness = max(1, int(font_scale * 2))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Position text at bottom right with some padding
    text_x = width - text_size[0] - 10
    text_y = height - 10
    
    # Add text with a slight shadow for better visibility
    cv2.putText(overlay, text, (text_x+2, text_y+2), font, font_scale, (0, 0, 0, 128), thickness)
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255, 200), thickness)
    
    # Apply the overlay
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Save the watermarked image
    watermarked_path = os.path.splitext(image_path)[0] + "_watermarked" + os.path.splitext(image_path)[1]
    cv2.imwrite(watermarked_path, img)
    
    return watermarked_path

def log_access(user_id, event_id, cluster_id, ip_address, db_connection):
    """Log user access to a gallery"""
    cursor = db_connection.cursor()
    cursor.execute(
        "INSERT INTO access_logs (user_id, event_id, cluster_id, ip_address) VALUES (?, ?, ?, ?)",
        (user_id, event_id, cluster_id, ip_address)
    )
    db_connection.commit()