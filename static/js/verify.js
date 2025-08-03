/**
 * FaceSnap Gallery - Verification Page JavaScript
 * Handles selfie capture and verification process
 */

let stream = null;
let capturedImage = null;

document.addEventListener('DOMContentLoaded', function() {
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const retakeBtn = document.getElementById('retakeBtn');
    const fileUploadBtn = document.getElementById('fileUploadBtn');
    const selfieInput = document.getElementById('selfieInput');
    const selfiePreview = document.getElementById('selfiePreview');
    const videoContainer = document.getElementById('videoContainer');
    const previewContainer = document.getElementById('previewContainer');
    const verifyForm = document.getElementById('verifyForm');
    const submitBtn = document.getElementById('submitBtn');
    const uploadTab = document.getElementById('upload-tab');
    const cameraTab = document.getElementById('camera-tab');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    // Initialize the verification form
    initVerificationForm();
    
    // Start camera when camera tab is clicked
    cameraTab.addEventListener('click', function() {
        startCamera();
    });
    
    // Handle file upload tab click
    uploadTab.addEventListener('click', function() {
        stopCamera();
    });
    
    // Capture button click
    captureBtn.addEventListener('click', function() {
        captureSelfie();
    });
    
    // Retake button click
    retakeBtn.addEventListener('click', function() {
        retakeSelfie();
    });
    
    // File input change
    selfieInput.addEventListener('change', function(e) {
        handleFileSelect(e);
    });
    
    // File upload button click
    fileUploadBtn.addEventListener('click', function() {
        selfieInput.click();
    });
    
    // Form submission
    verifyForm.addEventListener('submit', function(e) {
        e.preventDefault();
        submitVerification();
    });
    
    /**
     * Initialize the verification form
     */
    function initVerificationForm() {
        // Start with camera if available
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            startCamera();
        } else {
            // Camera not available, switch to upload tab
            const uploadTabInstance = new bootstrap.Tab(uploadTab);
            uploadTabInstance.show();
            document.getElementById('camera-warning').style.display = 'block';
        }
        
        // Initialize form validation
        validateForm();
    }
    
    /**
     * Start the camera stream
     */
    function startCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Stop any existing stream
            stopCamera();
            
            // Request camera access
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(mediaStream) {
                    stream = mediaStream;
                    videoElement.srcObject = mediaStream;
                    videoElement.play();
                    videoContainer.style.display = 'block';
                    previewContainer.style.display = 'none';
                    captureBtn.style.display = 'block';
                    retakeBtn.style.display = 'none';
                    updateSubmitButtonState();
                })
                .catch(function(err) {
                    console.error('Error accessing camera: ', err);
                    document.getElementById('camera-error').style.display = 'block';
                    document.getElementById('camera-error').textContent = 'Error accessing camera: ' + err.message;
                    
                    // Switch to upload tab
                    const uploadTabInstance = new bootstrap.Tab(uploadTab);
                    uploadTabInstance.show();
                });
        }
    }
    
    /**
     * Stop the camera stream
     */
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => {
                track.stop();
            });
            stream = null;
            videoElement.srcObject = null;
        }
    }
    
    /**
     * Capture a selfie from the video stream
     */
    function captureSelfie() {
        if (!stream) return;
        
        const context = canvasElement.getContext('2d');
        const width = videoElement.videoWidth;
        const height = videoElement.videoHeight;
        
        // Set canvas dimensions to match video
        canvasElement.width = width;
        canvasElement.height = height;
        
        // Draw video frame to canvas
        context.drawImage(videoElement, 0, 0, width, height);
        
        // Get image data URL
        capturedImage = canvasElement.toDataURL('image/jpeg');
        
        // Display the captured image
        selfiePreview.src = capturedImage;
        videoContainer.style.display = 'none';
        previewContainer.style.display = 'block';
        captureBtn.style.display = 'none';
        retakeBtn.style.display = 'block';
        
        // Update hidden input with image data
        document.getElementById('selfieData').value = capturedImage;
        
        // Update submit button state
        updateSubmitButtonState();
    }
    
    /**
     * Retake the selfie
     */
    function retakeSelfie() {
        capturedImage = null;
        document.getElementById('selfieData').value = '';
        selfiePreview.src = '';
        videoContainer.style.display = 'block';
        previewContainer.style.display = 'none';
        captureBtn.style.display = 'block';
        retakeBtn.style.display = 'none';
        
        // Restart camera if needed
        if (!stream || !stream.active) {
            startCamera();
        }
        
        // Update submit button state
        updateSubmitButtonState();
    }
    
    /**
     * Handle file selection for selfie upload
     */
    function handleFileSelect(e) {
        const file = e.target.files[0];
        
        if (file) {
            // Check if file is an image
            if (!file.type.match('image.*')) {
                alert('Please select an image file.');
                return;
            }
            
            // Check file size (max 5MB)
            if (file.size > 5 * 1024 * 1024) {
                alert('File size too large. Please select an image under 5MB.');
                return;
            }
            
            const reader = new FileReader();
            
            reader.onload = function(e) {
                capturedImage = e.target.result;
                selfiePreview.src = capturedImage;
                videoContainer.style.display = 'none';
                previewContainer.style.display = 'block';
                
                // Update hidden input with image data
                document.getElementById('selfieData').value = capturedImage;
                
                // Update submit button state
                updateSubmitButtonState();
            };
            
            reader.readAsDataURL(file);
        }
    }
    
    /**
     * Validate the form inputs
     */
    function validateForm() {
        const nameInput = document.getElementById('name');
        const emailInput = document.getElementById('email');
        const phoneInput = document.getElementById('phone');
        
        // Add input event listeners for validation
        [nameInput, emailInput, phoneInput].forEach(input => {
            input.addEventListener('input', function() {
                updateSubmitButtonState();
            });
        });
    }
    
    /**
     * Update the submit button state based on form validity
     */
    function updateSubmitButtonState() {
        const nameInput = document.getElementById('name');
        const emailInput = document.getElementById('email');
        const phoneInput = document.getElementById('phone');
        
        // Check if we have a selfie (either captured or uploaded)
        const hasSelfie = capturedImage !== null || document.getElementById('selfieData').value !== '';
        
        // Check if name is provided
        const hasName = nameInput.value.trim() !== '';
        
        // Check if either email or phone is provided
        const hasContact = emailInput.value.trim() !== '' || phoneInput.value.trim() !== '';
        
        // Enable submit button if all required fields are filled
        submitBtn.disabled = !(hasSelfie && hasName && hasContact);
    }
    
    /**
     * Submit the verification form
     */
    function submitVerification() {
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        submitBtn.disabled = true;
        
        // Create FormData
        const formData = new FormData(verifyForm);
        
        // Create and configure XMLHttpRequest
        const xhr = new XMLHttpRequest();
        xhr.open('POST', verifyForm.action, true);
        
        // Handle response
        xhr.onload = function() {
            loadingIndicator.style.display = 'none';
            
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    
                    if (response.success) {
                        // Redirect to gallery
                        window.location.href = response.redirect;
                    } else {
                        // Show error message
                        alert(response.error || 'Verification failed. Please try again.');
                        submitBtn.disabled = false;
                    }
                } catch (e) {
                    // Invalid JSON response
                    console.error('Invalid response:', e);
                    alert('An error occurred. Please try again.');
                    submitBtn.disabled = false;
                }
            } else {
                // HTTP error
                alert('Server error: ' + xhr.status);
                submitBtn.disabled = false;
            }
        };
        
        // Handle network errors
        xhr.onerror = function() {
            loadingIndicator.style.display = 'none';
            alert('Network error. Please check your connection and try again.');
            submitBtn.disabled = false;
        };
        
        // Send the form data
        xhr.send(formData);
    }
    
    // Clean up when page is unloaded
    window.addEventListener('beforeunload', function() {
        stopCamera();
    });
});