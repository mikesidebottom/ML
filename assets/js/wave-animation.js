/**
 * Wave animation using vanilla JavaScript
 * Creates a dots-based wave animation like the one in 0001.png
 * Disabled on notebook pages to prevent performance issues
 */

// Wait for DOM to load before initializing
document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a notebook page - if so, don't initialize the animation
  if (document.querySelector('.notebook-container')) {
    console.log('Notebook page detected - wave animation disabled');
    return;
  }

  // Get reference to the canvas element
  const container = document.querySelector('.wave-animation-container');
  if (!container) return;
  
  // Create canvas element manually instead of using p5.js
  const canvas = document.createElement('canvas');
  canvas.width = container.offsetWidth;
  canvas.height = container.offsetHeight;
  document.getElementById('waveCanvas').appendChild(canvas);
  
  // Get 2D context for drawing
  const ctx = canvas.getContext('2d');
  let time = 0;
  
  // Constants for animation
  const bgColor = '#1F1F1F';
  const waveCount = 25;
  const dotSpacing = 6;
  
  // Simple noise function (replacement for p5.noise)
  function noise(x, y) {
    return 0.5 * (
      Math.sin(x * 0.1 + time * 0.1) + 
      Math.sin(y * 0.1 + time * 0.2) +
      Math.sin((x + y) * 0.05) + 
      Math.sin(Math.sqrt(x*x + y*y) * 0.1)
    ) + 0.5;
  }
  
  // Handle window resize
  window.addEventListener('resize', function() {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
  });
  
  // Animation function
  function animate() {
    // Clear the canvas
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Position for waves at the bottom of the screen
    const baseY = canvas.height * 0.7;
    
    // Draw multiple wave layers
    for (let i = 0; i < waveCount; i++) {
      // Calculate color based on layer position (brighter at front)
      const brightness = Math.floor(150 + (i / waveCount) * 105);
      const opacity = 0.2 + (i / waveCount) * 0.8;
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Draw wave as series of dots
      for (let x = 0; x < canvas.width; x += dotSpacing) {
        // Calculate y position using noise function
        const waveHeight = 80 * ((waveCount - i) / waveCount);
        const y = baseY - waveHeight * noise(x * 0.01, i * 0.5 + time);
        
        // Draw dot with size based on layer position
        const dotSize = 1 + (i / waveCount) * 2;
        ctx.beginPath();
        ctx.arc(x, y, dotSize, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Update time for next animation frame
    time += 0.03;
    requestAnimationFrame(animate);
  }
  
  // Start the animation
  animate();
});