
document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');
    const recognizeButton = document.getElementById('recognizeButton');
    const predictionElement = document.getElementById('prediction');
    
    // Backend API endpoint
    const apiUrl = 'http://localhost:5000/predict';
    
    // Setup canvas
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    
    // Variables for drawing
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch support for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Button event listeners
    clearButton.addEventListener('click', clearCanvas);
    recognizeButton.addEventListener('click', recognizeDigit);
    
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const offsetX = touch.clientX - rect.left;
        const offsetY = touch.clientY - rect.top;
        
        isDrawing = true;
        [lastX, lastY] = [offsetX, offsetY];
    }
    
    function handleTouchMove(e) {
        if (!isDrawing) return;
        e.preventDefault();
        
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const offsetX = touch.clientX - rect.left;
        const offsetY = touch.clientY - rect.top;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(offsetX, offsetY);
        ctx.stroke();
        
        [lastX, lastY] = [offsetX, offsetY];
    }
    
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictionElement.textContent = '-';
    }
    
    function recognizeDigit() {
        // Convert canvas to base64 image
        const imageData = canvas.toDataURL('image/png');
        
        // Show loading state
        predictionElement.textContent = 'Processing...';
        
        // Send to backend API
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                predictionElement.textContent = data.prediction;
            } else {
                predictionElement.textContent = 'Error';
                console.error('Error:', data.error);
            }
        })
        .catch(error => {
            predictionElement.textContent = 'Error';
            console.error('Error:', error);
        });
    }
    
    // Initialize with a clear canvas
    clearCanvas();
});