/**
 * FaceSnap Gallery - File Upload Handler
 * Provides drag and drop file upload functionality with image previews
 */

/**
 * Setup file upload with drag and drop functionality
 * @param {string} dropAreaId - ID of the drop area element
 * @param {string} fileInputId - ID of the file input element
 * @param {string} previewContainerId - ID of the preview container element
 * @param {Function} onFilesChanged - Callback function when files change
 * @returns {Object} - Object with methods to interact with the uploader
 */
function setupFileUpload(dropAreaId, fileInputId, previewContainerId, onFilesChanged) {
    const dropArea = document.getElementById(dropAreaId);
    const fileInput = document.getElementById(fileInputId);
    const previewContainer = document.getElementById(previewContainerId);
    
    // Maximum number of files to preview
    const MAX_PREVIEW_FILES = 20;
    // Maximum file size in bytes (50MB)
    const MAX_FILE_SIZE = 50 * 1024 * 1024;
    // Allowed file types
    const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/heic', 'image/heif'];
    
    // Initialize
    init();
    
    /**
     * Initialize the file upload functionality
     */
    function init() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        // Remove highlight when item is dragged out or dropped
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle dropped files
        dropArea.addEventListener('drop', handleDrop, false);
        
        // Handle file input change
        fileInput.addEventListener('change', handleFileInputChange, false);
        
        // Handle click on drop area
        dropArea.addEventListener('click', () => {
            fileInput.click();
        });
    }
    
    /**
     * Prevent default drag and drop behaviors
     * @param {Event} e - The event object
     */
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    /**
     * Highlight the drop area
     */
    function highlight() {
        dropArea.classList.add('active');
    }
    
    /**
     * Remove highlight from the drop area
     */
    function unhighlight() {
        dropArea.classList.remove('active');
    }
    
    /**
     * Handle dropped files
     * @param {Event} e - The drop event
     */
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    /**
     * Handle file input change
     * @param {Event} e - The change event
     */
    function handleFileInputChange(e) {
        const files = e.target.files;
        handleFiles(files);
    }
    
    /**
     * Process the selected files
     * @param {FileList} files - The list of files
     */
    function handleFiles(files) {
        // Filter files
        const validFiles = Array.from(files).filter(file => {
            // Check file type
            if (!ALLOWED_TYPES.includes(file.type)) {
                console.warn(`File type not allowed: ${file.type}`);
                return false;
            }
            
            // Check file size
            if (file.size > MAX_FILE_SIZE) {
                console.warn(`File too large: ${formatFileSize(file.size)}`);
                return false;
            }
            
            return true;
        });
        
        // Update file input with valid files
        updateFileInput(validFiles);
        
        // Create previews
        createPreviews(validFiles);
        
        // Call the callback function
        if (typeof onFilesChanged === 'function') {
            onFilesChanged(validFiles);
        }
    }
    
    /**
     * Update the file input with valid files
     * @param {Array} validFiles - Array of valid files
     */
    function updateFileInput(validFiles) {
        // Create a new DataTransfer object
        const dataTransfer = new DataTransfer();
        
        // Add valid files to the DataTransfer object
        validFiles.forEach(file => {
            dataTransfer.items.add(file);
        });
        
        // Set the file input files
        fileInput.files = dataTransfer.files;
    }
    
    /**
     * Create image previews for the selected files
     * @param {Array} files - Array of files
     */
    function createPreviews(files) {
        // Clear previous previews
        previewContainer.innerHTML = '';
        
        // Show preview container if there are files
        if (files.length > 0) {
            previewContainer.style.display = 'grid';
        } else {
            previewContainer.style.display = 'none';
            return;
        }
        
        // Limit the number of previews
        const filesToPreview = files.slice(0, MAX_PREVIEW_FILES);
        
        // Create preview elements
        filesToPreview.forEach((file, index) => {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                const preview = document.createElement('div');
                preview.className = 'preview-item';
                preview.dataset.index = index;
                
                // Create image element
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = file.name;
                
                // Create info overlay
                const info = document.createElement('div');
                info.className = 'preview-info';
                
                // File name
                const name = document.createElement('div');
                name.className = 'preview-name';
                name.textContent = truncateFilename(file.name, 20);
                
                // File size
                const size = document.createElement('div');
                size.className = 'preview-size';
                size.textContent = formatFileSize(file.size);
                
                // Remove button
                const removeBtn = document.createElement('button');
                removeBtn.className = 'preview-remove';
                removeBtn.innerHTML = '<i class="fas fa-times"></i>';
                removeBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    removeFile(index);
                });
                
                // Append elements
                info.appendChild(name);
                info.appendChild(size);
                preview.appendChild(img);
                preview.appendChild(info);
                preview.appendChild(removeBtn);
                previewContainer.appendChild(preview);
            };
            
            reader.readAsDataURL(file);
        });
        
        // Show message if there are more files than the preview limit
        if (files.length > MAX_PREVIEW_FILES) {
            const message = document.createElement('div');
            message.className = 'preview-more';
            message.textContent = `+${files.length - MAX_PREVIEW_FILES} more files`;
            previewContainer.appendChild(message);
        }
    }
    
    /**
     * Remove a file from the selection
     * @param {number} index - Index of the file to remove
     */
    function removeFile(index) {
        // Get current files
        const currentFiles = Array.from(fileInput.files);
        
        // Remove the file at the specified index
        const updatedFiles = currentFiles.filter((_, i) => i !== index);
        
        // Update file input and previews
        updateFileInput(updatedFiles);
        createPreviews(updatedFiles);
        
        // Call the callback function
        if (typeof onFilesChanged === 'function') {
            onFilesChanged(updatedFiles);
        }
    }
    
    /**
     * Clear all selected files
     */
    function clearFiles() {
        // Clear file input
        fileInput.value = '';
        
        // Clear previews
        previewContainer.innerHTML = '';
        previewContainer.style.display = 'none';
        
        // Call the callback function
        if (typeof onFilesChanged === 'function') {
            onFilesChanged([]);
        }
    }
    
    /**
     * Format file size in human-readable format
     * @param {number} bytes - File size in bytes
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
     * Truncate filename if it's too long
     * @param {string} filename - The filename to truncate
     * @param {number} maxLength - Maximum length before truncation
     * @returns {string} - Truncated filename
     */
    function truncateFilename(filename, maxLength) {
        if (filename.length <= maxLength) return filename;
        
        const extension = filename.split('.').pop();
        const nameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
        
        // Calculate how many characters we can keep
        const charsToKeep = maxLength - extension.length - 3; // 3 for '...' and '.' before extension
        
        if (charsToKeep <= 0) return '...' + extension;
        
        return nameWithoutExt.substring(0, charsToKeep) + '...' + '.' + extension;
    }
    
    // Return public methods
    return {
        clearFiles: clearFiles
    };
}