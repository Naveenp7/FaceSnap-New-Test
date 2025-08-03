/**
 * FaceSnap Gallery - Gallery Page JavaScript
 * Handles gallery interactions, sharing, and image viewing
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize lightbox
    initLightbox();
    
    // Initialize share buttons
    initShareButtons();
    
    // Initialize copy URL functionality
    initCopyUrl();
});

/**
 * Initialize Lightbox for image viewing
 */
function initLightbox() {
    lightbox.option({
        'resizeDuration': 200,
        'wrapAround': true,
        'albumLabel': 'Photo %1 of %2',
        'fadeDuration': 300,
        'imageFadeDuration': 300
    });
}

/**
 * Initialize share URL copy functionality
 */
function initCopyUrl() {
    const copyBtn = document.getElementById('copyShareUrlBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            copyShareUrl();
        });
    }
}

/**
 * Copy share URL to clipboard
 */
function copyShareUrl() {
    const shareUrl = document.getElementById('shareUrl');
    if (!shareUrl) return;
    
    shareUrl.select();
    document.execCommand('copy');
    
    // Show a temporary tooltip
    const button = document.getElementById('copyShareUrlBtn');
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
    
    setTimeout(function() {
        button.innerHTML = originalText;
    }, 2000);
}

/**
 * Initialize share buttons
 */
function initShareButtons() {
    // Add event listeners to share buttons if they exist
    const shareButtons = document.querySelectorAll('[data-share-type]');
    shareButtons.forEach(button => {
        button.addEventListener('click', function() {
            const shareType = this.getAttribute('data-share-type');
            shareVia(shareType);
        });
    });
}

/**
 * Share image via different platforms
 * @param {string} platform - The platform to share on (email, whatsapp, facebook, pinterest)
 */
function shareVia(platform) {
    const shareUrl = document.getElementById('shareUrl').value;
    let shareLink = '';
    
    switch(platform) {
        case 'email':
            shareLink = `mailto:?subject=Check out this photo&body=I wanted to share this photo with you: ${shareUrl}`;
            break;
        case 'whatsapp':
            shareLink = `https://wa.me/?text=${encodeURIComponent('Check out this photo: ' + shareUrl)}`;
            break;
        case 'facebook':
            shareLink = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
            break;
        case 'pinterest':
            shareLink = `https://pinterest.com/pin/create/button/?url=${encodeURIComponent(shareUrl)}&media=${encodeURIComponent(shareUrl)}&description=${encodeURIComponent('Check out this photo')}`;
            break;
    }
    
    if (shareLink) {
        window.open(shareLink, '_blank');
    }
}

/**
 * Open share modal for a specific image
 * @param {string} url - The URL of the image to share
 */
function shareImage(url) {
    document.getElementById('shareUrl').value = url;
    const shareModal = new bootstrap.Modal(document.getElementById('shareModal'));
    shareModal.show();
}

/**
 * Download all images in the gallery
 * @param {string} downloadUrl - The URL to download all images
 */
function downloadAllImages(downloadUrl) {
    // Show loading indicator
    const downloadBtn = document.querySelector('.download-all-btn');
    if (downloadBtn) {
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Preparing...';
        downloadBtn.disabled = true;
        
        // Start download
        window.location.href = downloadUrl;
        
        // Reset button after a delay
        setTimeout(function() {
            downloadBtn.innerHTML = originalText;
            downloadBtn.disabled = false;
        }, 3000);
    }
}