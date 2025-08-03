import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from urllib.parse import urlparse

# Configure Cloudinary
cloudinary.config(
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME', 'dwqajqcnq'),
    api_key = os.getenv('CLOUDINARY_API_KEY', '343324552376533'),
    api_secret = os.getenv('CLOUDINARY_API_SECRET', 'Zsfm4DL6-VX4siB4xCNvi-XiTZw'),
    secure = True
)

def upload_file(file_path, folder="uploads"):
    """Upload a file to Cloudinary"""
    try:
        # Upload the file
        result = cloudinary.uploader.upload(
            file_path,
            folder=folder,
            resource_type="auto"
        )
        return result['secure_url']
    except Exception as e:
        print(f"Error uploading to Cloudinary: {str(e)}")
        return None

def upload_image_data(image_data, folder="uploads"):
    """Upload image data to Cloudinary"""
    try:
        # Upload the image data
        result = cloudinary.uploader.upload(
            image_data,
            folder=folder,
            resource_type="auto"
        )
        return result['secure_url']
    except Exception as e:
        print(f"Error uploading to Cloudinary: {str(e)}")
        return None

def delete_file(public_id):
    """Delete a file from Cloudinary"""
    try:
        result = cloudinary.uploader.destroy(public_id)
        return result['result'] == 'ok'
    except Exception as e:
        print(f"Error deleting from Cloudinary: {str(e)}")
        return False

def get_public_id_from_url(url):
    """Extract public_id from Cloudinary URL"""
    try:
        path = urlparse(url).path
        # Remove version number if present
        filename = path.split('/')[-1]
        # Remove extension
        public_id = os.path.splitext(filename)[0]
        return public_id
    except:
        return None
