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
  const waveCount = 35; // Increased for more complexity
  const dotSpacing = 4; // Smaller spacing for more detail
  const amplitude = 40; // Maximum height of waves
  const waveFreq = 0.015; // Wave frequency
  const speed = 0.02; // Animation speed
  
  // Simple noise function (replacement for p5.noise)
  function noise(x, y) {
    // Using multiple sine waves at different frequencies creates more natural patterns
    return (
      Math.sin(x * waveFreq + time) * 0.5 + 
      Math.sin(x * waveFreq * 0.5 + time * 1.3) * 0.3 +
      Math.sin(x * waveFreq * 0.2 + time * 0.7 + y) * 0.2
    ) * 0.5 + 0.5; // Normalize to 0..1
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
    const baseY = canvas.height * 0.75; // Lower position for better visibility
    
    // Draw multiple wave layers from back to front
    for (let i = 0; i < waveCount; i++) {
      // Calculate layer parameters
      const layerDepth = i / waveCount; // 0 = back, 1 = front
      const layerAmplitude = amplitude * (0.3 + layerDepth * 0.7); // Larger amplitude in front
      
      // Calculate color based on layer position (brighter in front)
      const brightness = Math.floor(100 + layerDepth * 155); // 100-255
      const opacity = 0.1 + layerDepth * 0.9; // More opaque in front
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Draw dots with varying sizes
      const dotSize = 1 + layerDepth * 2; // Larger dots in front
      const xOffset = i * 7; // Offset each wave horizontally
      
      // Draw wave as series of dots
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        // Multiple overlapping sine waves for complex patterns
        const phase = time * (1 + layerDepth * 0.5); // Different speeds for each layer
        const noiseVal = noise(x, i * 5);
        
        // Calculate y position with combination of effects
        const y = baseY - 
                 layerAmplitude * noiseVal + // Main wave pattern
                 Math.sin(x * 0.03 + phase) * layerAmplitude * 0.3 + // Secondary wave
                 (1 - layerDepth) * 30; // Make back waves higher
        
        // Draw dot
        ctx.beginPath();
        ctx.arc(x, y, dotSize, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Update time for next animation frame
    time += speed;
    requestAnimationFrame(animate);
  }
  
  // Start the animation
  animate();
});