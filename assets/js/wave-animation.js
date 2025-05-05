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
  const waveCount = 50; // Increased for more layers
  const dotSpacing = 3; // Smaller spacing for more detail
  const amplitude = 120; // Significantly increased for larger waves
  const baseWaveFreq = 0.008; // Base frequency for waves
  const speed = 0.012; // Slightly reduced for smoother animation
  
  // Random offset for each wave to increase diversity
  const randomOffsets = Array.from({length: waveCount}, () => ({
    freq: (Math.random() * 0.01) + 0.005,  // Random frequency modifier
    phase: Math.random() * Math.PI * 2,    // Random phase offset
    amp: (Math.random() * 0.4) + 0.8,      // Random amplitude modifier (0.8-1.2)
    speed: (Math.random() * 0.5) + 0.75    // Random speed modifier (0.75-1.25)
  }));
  
  // Enhanced noise function with more randomization
  function noise(x, y, i) {
    // Get wave-specific randomizations
    const offset = randomOffsets[i % randomOffsets.length];
    const waveFreq = baseWaveFreq * offset.freq;
    const wavePhase = offset.phase;
    const waveTime = time * offset.speed;
    
    // Using multiple sine waves at different frequencies creates more natural patterns
    return (
      Math.sin(x * waveFreq + waveTime + wavePhase) * 0.5 + 
      Math.sin(x * waveFreq * 0.3 + waveTime * 1.3 + wavePhase) * 0.3 +
      Math.sin(x * waveFreq * 0.15 + waveTime * 0.7 + wavePhase + y) * 0.2
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
    const baseY = canvas.height * 0.38; // Moved slightly higher
    
    // Draw multiple wave layers from back to front
    for (let i = 0; i < waveCount; i++) {
      // Calculate layer parameters
      const layerDepth = i / waveCount; // 0 = back, 1 = front
      const offset = randomOffsets[i % randomOffsets.length];
      
      // Larger and more varied amplitude between layers
      const layerAmplitude = amplitude * (0.3 + layerDepth * 0.7) * offset.amp;
      
      // Calculate color based on layer position (brighter in front)
      const brightness = Math.floor(110 + layerDepth * 145); // Brightened slightly
      const opacity = 0.1 + layerDepth * 0.9; // More contrast between layers
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Draw dots with varying sizes - bigger dots now
      const dotSize = 1.8 + layerDepth * 3.5; // Larger dots overall
      const xOffset = i * 12; // Larger offset for more horizontal variation
      
      // Draw wave as series of dots
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        // Multiple overlapping sine waves for complex patterns
        const phase = time * (1 + layerDepth * 0.7) * offset.speed;
        const noiseVal = noise(x, i * 7, i); // Passing i to have wave-specific noise
        
        // Calculate y position with combination of effects - more variation
        const y = baseY - 
                 layerAmplitude * noiseVal + // Main wave pattern 
                 Math.sin(x * 0.03 * offset.freq + phase) * layerAmplitude * 0.4 + // Secondary wave
                 Math.sin(x * 0.007 * offset.freq + phase * 0.5 + offset.phase) * layerAmplitude * 0.25 + // Tertiary wave
                 (1 - layerDepth) * 90; // Make back waves higher (increased)
        
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