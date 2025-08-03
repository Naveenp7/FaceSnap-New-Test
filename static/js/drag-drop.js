function initDragDrop(options) {
    const dropZone = document.getElementById(options.dropZoneId);
    const fileInput = document.getElementById(options.fileInputId);
    const previewContainer = document.getElementById(options.previewContainerId);
    const imagePreview = document.getElementById(options.imagePreviewId);

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        // Update the file input with the dropped files
        const dT = new DataTransfer();
        for(let file of files) {
            dT.items.add(file);
        }
        fileInput.files = dT.files;
        
        // Trigger the change event to update preview
        fileInput.dispatchEvent(new Event('change'));
    }

    function previewFiles(files) {
        imagePreview.innerHTML = '';
        
        if (files.length > 0) {
            previewContainer.style.display = 'block';
            
            for (let i = 0; i < Math.min(files.length, 30); i++) {
                const file = files[i];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const col = document.createElement('div');
                    col.className = 'col';
                    
                    const card = document.createElement('div');
                    card.className = 'card h-100';
                    
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'card-img-top';
                    img.alt = 'Preview';
                    
                    const cardBody = document.createElement('div');
                    cardBody.className = 'card-body p-2';
                    
                    const fileName = document.createElement('p');
                    fileName.className = 'card-text small text-truncate mb-0';
                    fileName.textContent = file.name;
                    
                    cardBody.appendChild(fileName);
                    card.appendChild(img);
                    card.appendChild(cardBody);
                    col.appendChild(card);
                    imagePreview.appendChild(col);
                };
                
                reader.readAsDataURL(file);
            }
            
            if (files.length > 20) {
                const col = document.createElement('div');
                col.className = 'col';
                
                const card = document.createElement('div');
                card.className = 'card h-100 d-flex justify-content-center align-items-center';
                
                const cardBody = document.createElement('div');
                cardBody.className = 'card-body text-center';
                
                const moreText = document.createElement('p');
                moreText.className = 'mb-0 fw-bold';
                moreText.textContent = `+${files.length - 20} more`;
                
                cardBody.appendChild(moreText);
                card.appendChild(cardBody);
                col.appendChild(card);
                imagePreview.appendChild(col);
            }
        } else {
            previewContainer.style.display = 'none';
        }
    }

    // Handle file input change
    fileInput.addEventListener('change', function() {
        previewFiles(this.files);
    });
}
