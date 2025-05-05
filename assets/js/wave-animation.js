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
  
  // Constants for animation - increased for more impact
  const bgColor = '#1F1F1F';
  const waveCount = 60; // Increased for more layers
  const dotSpacing = 4; // More spacing between dots to make lines thinner
  const amplitude = 150; // Significantly increased amplitude for larger waves
  const baseWaveFreq = 0.01; // Higher base frequency for waves
  const speed = 0.018; // Increased for faster animation
  
  // More varied random offset for each wave to increase diversity
  const randomOffsets = Array.from({length: waveCount}, () => ({
    freq: (Math.random() * 0.02) + 0.005,  // More varied frequency modifier
    phase: Math.random() * Math.PI * 4,    // Larger phase offset range
    amp: (Math.random() * 0.6) + 0.7,      // Wider amplitude variation (0.7-1.3)
    speed: (Math.random() * 0.8) + 0.7,    // Wider speed variation (0.7-1.5)
    // Add initial chaos factors to make it interesting from the start
    initialX: Math.random() * 1000,
    initialY: Math.random() * 1000,
    turbulence: (Math.random() * 0.5) + 0.5 // Turbulence factor
  }));
  
  // Enhanced noise function with more initial randomization
  function noise(x, y, i) {
    // Get wave-specific randomizations
    const offset = randomOffsets[i % randomOffsets.length];
    const waveFreq = baseWaveFreq * offset.freq;
    const wavePhase = offset.phase;
    const waveTime = time * offset.speed;
    
    // Add turbulence to make waves more chaotic from the start
    const turbX = x + offset.initialX;
    const turbY = y + offset.initialY;
    const turbulenceFactor = offset.turbulence;
    
    // Using multiple sine waves with different frequencies and more chaos
    return (
      Math.sin(turbX * waveFreq + waveTime + wavePhase) * 0.4 + 
      Math.cos(turbX * waveFreq * 0.5 + waveTime * 1.3 + wavePhase) * 0.3 +
      Math.sin(turbX * waveFreq * 0.2 + turbY * 0.05 + waveTime * 0.7) * 0.2 +
      // Add perlin-like noise approximation for more organic feel
      Math.sin(turbX * 0.01 * turbulenceFactor + turbY * 0.01) * 
      Math.cos(turbY * 0.01 * turbulenceFactor + waveTime) * 0.1
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
    
    // Position for waves higher up on the screen (0.5 is middle, lower values = higher position)
    const baseY = canvas.height * 0.35; // Moved even higher
    
    // Draw multiple wave layers from back to front
    for (let i = 0; i < waveCount; i++) {
      // Calculate layer parameters
      const layerDepth = i / waveCount; // 0 = back, 1 = front
      const offset = randomOffsets[i % randomOffsets.length];
      
      // Larger and more varied amplitude between layers
      const layerAmplitude = amplitude * (0.3 + layerDepth * 0.7) * offset.amp;
      
      // Calculate color based on layer position (brighter in front)
      const brightness = Math.floor(120 + layerDepth * 135); // Adjusted brightness
      const opacity = 0.08 + layerDepth * 0.92; // More transparency in the back layers
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Draw smaller dots with varying sizes for thinner lines
      const dotSize = 1.2 + layerDepth * 2.8; // Smaller dots overall
      const xOffset = i * 15; // Larger horizontal offset for more variation
      
      // Draw wave as series of dots
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        // Multiple overlapping waves for complex patterns
        const phase = time * (1 + layerDepth * 0.8) * offset.speed;
        const noiseVal = noise(x, i * 9, i); // Passing i to have wave-specific noise
        
        // Calculate y position with more chaotic effects
        const y = baseY - 
                 layerAmplitude * noiseVal + // Main wave pattern 
                 Math.sin(x * 0.04 * offset.freq + phase) * layerAmplitude * 0.5 + // Secondary wave - increased effect
                 Math.sin(x * 0.009 * offset.freq + phase * 0.7 + offset.phase) * layerAmplitude * 0.35 + // Tertiary wave - increased effect
                 (1 - layerDepth) * 110; // Make back waves higher (increased)
        
        // Draw dot
        ctx.beginPath();
        ctx.arc(x, y, dotSize, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Update time for next animation frame - faster animation
    time += speed;
    requestAnimationFrame(animate);
  }
  
  // Start the animation
  animate();
});