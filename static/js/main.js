/**
 * FaceSnap Gallery - Main JavaScript
 * Contains common functionality used across the application
 */

// Initialize all tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Handle flash messages auto-dismiss
    const flashMessages = document.querySelectorAll('.alert-dismissible.auto-dismiss');
    flashMessages.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000); // Auto dismiss after 5 seconds
    });
});

/**
 * Copy text to clipboard
 * @param {string} text - The text to copy
 * @param {HTMLElement} button - The button element that triggered the copy
 * @param {string} successText - Text to show on success (optional)
 */
function copyToClipboard(text, button, successText = 'Copied!') {
    // Create a temporary input element
    const tempInput = document.createElement('input');
    tempInput.value = text;
    document.body.appendChild(tempInput);
    tempInput.select();
    document.execCommand('copy');
    document.body.removeChild(tempInput);
    
    // Show success feedback
    const originalText = button.innerHTML;
    button.innerHTML = `<i class="fas fa-check"></i> ${successText}`;
    button.disabled = true;
    
    // Reset button after 2 seconds
    setTimeout(() => {
        button.innerHTML = originalText;
        button.disabled = false;
    }, 2000);
}

/**
 * Format file size in human-readable format
 * @param {number} bytes - The file size in bytes
 * @returns {string} - Formatted file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Create a file upload preview with drag and drop support
 * @param {string} dropAreaId - The ID of the drop area element
 * @param {string} fileInputId - The ID of the file input element
 * @param {string} previewContainerId - The ID of the preview container element
 * @param {function} onFileChange - Callback function when files change
 */
function setupFileUpload(dropAreaId, fileInputId, previewContainerId, onFileChange) {
    const dropArea = document.getElementById(dropAreaId);
    const fileInput = document.getElementById(fileInputId);
    const previewContainer = document.getElementById(previewContainerId);
    
    if (!dropArea || !fileInput || !previewContainer) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Handle file input change
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // Click on drop area should trigger file input
    dropArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropArea.classList.add('dragover');
    }
    
    function unhighlight() {
        dropArea.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    function handleFiles(files) {
        // Update file input with the dropped files
        if (files.length > 0) {
            // Create a new DataTransfer object
            const dataTransfer = new DataTransfer();
            
            // Add existing files from the input
            if (fileInput.files) {
                Array.from(fileInput.files).forEach(file => {
                    dataTransfer.items.add(file);
                });
            }
            
            // Add new files
            Array.from(files).forEach(file => {
                // Only add image files
                if (file.type.startsWith('image/')) {
                    dataTransfer.items.add(file);
                }
            });
            
            // Set the new files to the input
            fileInput.files = dataTransfer.files;
            
            // Update preview
            updatePreview();
            
            // Call the callback if provided
            if (typeof onFileChange === 'function') {
                onFileChange(fileInput.files);
            }
        }
    }
    
    function updatePreview() {
        // Clear preview
        previewContainer.innerHTML = '';
        
        if (fileInput.files.length > 0) {
            Array.from(fileInput.files).forEach((file, index) => {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const previewItem = document.createElement('div');
                    previewItem.className = 'preview-item';
                    previewItem.innerHTML = `
                        <img src="${e.target.result}" alt="Preview">
                        <div class="remove-btn" data-index="${index}" title="Remove">
                            <i class="fas fa-times"></i>
                        </div>
                        <div class="small text-center mt-1">${file.name.substring(0, 15)}${file.name.length > 15 ? '...' : ''}</div>
                    `;
                    previewContainer.appendChild(previewItem);
                    
                    // Add remove button event listener
                    const removeBtn = previewItem.querySelector('.remove-btn');
                    removeBtn.addEventListener('click', function() {
                        removeFile(parseInt(this.dataset.index));
                    });
                };
                
                reader.readAsDataURL(file);
            });
            
            // Show the preview container
            previewContainer.style.display = 'grid';
        } else {
            // Hide the preview container if no files
            previewContainer.style.display = 'none';
        }
    }
    
    function removeFile(index) {
        if (index >= 0 && index < fileInput.files.length) {
            const dataTransfer = new DataTransfer();
            
            Array.from(fileInput.files)
                .filter((_, i) => i !== index)
                .forEach(file => dataTransfer.items.add(file));
            
            fileInput.files = dataTransfer.files;
            updatePreview();
            
            // Call the callback if provided
            if (typeof onFileChange === 'function') {
                onFileChange(fileInput.files);
            }
        }
    }
    
    // Initial preview update
    updatePreview();
    
    // Return methods for external use
    return {
        addFiles: handleFiles,
        removeFile: removeFile,
        clearFiles: function() {
            fileInput.value = '';
            updatePreview();
            if (typeof onFileChange === 'function') {
                onFileChange(fileInput.files);
            }
        }
    };
}

/**
 * Format a date string in a more readable format
 * @param {string} dateString - The date string to format
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    const options = { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return date.toLocaleDateString(undefined, options);
}

/**
 * Show a confirmation dialog
 * @param {string} message - The confirmation message
 * @param {function} onConfirm - Callback function when confirmed
 * @param {string} confirmBtnText - Text for the confirm button
 * @param {string} cancelBtnText - Text for the cancel button
 */
function showConfirmDialog(message, onConfirm, confirmBtnText = 'Confirm', cancelBtnText = 'Cancel') {
    // Create modal elements
    const modalId = 'confirmModal' + Math.random().toString(36).substr(2, 9);
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = modalId;
    modal.setAttribute('tabindex', '-1');
    modal.setAttribute('aria-hidden', 'true');
    
    modal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Confirmation</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>${message}</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">${cancelBtnText}</button>
                    <button type="button" class="btn btn-primary" id="${modalId}-confirm">${confirmBtnText}</button>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to body
    document.body.appendChild(modal);
    
    // Initialize Bootstrap modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
    
    // Add confirm button event listener
    document.getElementById(`${modalId}-confirm`).addEventListener('click', function() {
        modalInstance.hide();
        if (typeof onConfirm === 'function') {
            onConfirm();
        }
    });
    
    // Remove modal from DOM after it's hidden
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
    });
}