document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictionDisplay = document.getElementById('prediction');
    const imageUpload = document.getElementById('imageUpload');
    
    let isDrawing = false;
    
    // Set up canvas for drawing
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    
    // Fill canvas with black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch support for mobile devices
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Button event listeners
    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predictDigit);
    
    // File upload event listener
    imageUpload.addEventListener('change', handleImageUpload);
    
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        // Get correct coordinates whether mouse or touch event
        const x = e.clientX || (e.touches && e.touches[0].clientX);
        const y = e.clientY || (e.touches && e.touches[0].clientY);
        
        if (x && y) {
            const rect = canvas.getBoundingClientRect();
            const canvasX = x - rect.left;
            const canvasY = y - rect.top;
            
            ctx.beginPath();
            ctx.moveTo(lastX || canvasX, lastY || canvasY);
            ctx.lineTo(canvasX, canvasY);
            ctx.lineWidth = 20;  
            ctx.strokeStyle = 'white';  
            ctx.stroke();
            
            lastX = canvasX;
            lastY = canvasY;
        }
    }
    
    function stopDrawing() {
        isDrawing = false;
        lastX = null;
        lastY = null;
    }
    
    function handleTouch(e) {
        e.preventDefault(); // Prevent scrolling while drawing
        
        if (e.type === 'touchstart') {
            startDrawing(e);
        } else if (e.type === 'touchmove') {
            draw(e);
        }
    }
    
    function clearCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionDisplay.textContent = '-';
    }
    
    function predictDigit() {
        // Display loading state
        predictionDisplay.textContent = '...';
        
        // Get image data from canvas
        const imageData = canvas.toDataURL('image/png');
        
        // Send to backend for prediction
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
                predictionDisplay.textContent = 'Error';
            } else {
                predictionDisplay.textContent = data.prediction;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            predictionDisplay.textContent = 'Error';
        });
    }
    
    function handleImageUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                // Clear canvas
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw uploaded image to canvas
                const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                const x = (canvas.width / 2) - (img.width / 2) * scale;
                const y = (canvas.height / 2) - (img.height / 2) * scale;
                
                ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
                
                // Predict automatically when image is uploaded
                predictDigit();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});
