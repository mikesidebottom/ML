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
  
  // Constants for animation - REDUCED wave count for better performance
  const bgColor = '#1F1F1F';
  const waveCount = 25; // REDUCED from 60 to 25 for better performance
  const dotSpacing = 5; // Slightly increased spacing for even less rendering
  const amplitude = 220; // Keep amplitude dramatic
  const baseWaveFreq = 0.018;
  const speed = 0.02;
  
  // Random offset for each wave - maintained for diversity
  const randomOffsets = Array.from({length: waveCount}, () => ({
    freq: (Math.random() * 0.03) + 0.008,
    phase: Math.random() * Math.PI * 6,
    amp: (Math.random() * 0.7) + 0.7,
    speed: (Math.random() * 0.9) + 0.5,
    initialX: Math.random() * 1000,
    initialY: Math.random() * 1000,
    turbulence: (Math.random() * 0.7) + 0.7
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
      Math.cos(turbX * waveFreq * 0.6 + waveTime * 1.5 + wavePhase) * 0.3 + // INCREASED frequency multiplier
      Math.sin(turbX * waveFreq * 0.3 + turbY * 0.08 + waveTime * 0.9) * 0.25 + // INCREASED values
      // Add perlin-like noise approximation for more organic feel
      Math.sin(turbX * 0.015 * turbulenceFactor + turbY * 0.015) * 
      Math.cos(turbY * 0.015 * turbulenceFactor + waveTime) * 0.15 // INCREASED influence
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
    const baseY = canvas.height * 0.33; // Moved even higher
    
    // Draw multiple wave layers from back to front
    for (let i = 0; i < waveCount; i++) {
      // Calculate layer parameters
      const layerDepth = i / waveCount; // 0 = back, 1 = front
      const offset = randomOffsets[i % randomOffsets.length];
      
      // Slightly increased individual wave amplitudes to compensate for fewer waves
      const layerAmplitude = amplitude * (0.35 + layerDepth * 0.75) * offset.amp;
      
      // Calculate color based on layer position (brighter in front)
      // Adjusted opacity to ensure good visibility with fewer layers
      const brightness = Math.floor(120 + layerDepth * 135);
      const opacity = 0.08 + layerDepth * 0.92; // Slightly increased minimum opacity
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Draw slightly larger dots to maintain visual impact with fewer waves
      const dotSize = 0.7 + layerDepth * 3.0; // Slightly increased dot size
      const xOffset = i * 22; // Increased offset for better horizontal distribution with fewer waves
      
      // Draw wave as series of dots
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        // Multiple overlapping waves for complex patterns
        const phase = time * (1 + layerDepth * 0.9) * offset.speed;
        const noiseVal = noise(x, i * 11, i); // INCREASED y factor for more vertical variation
        
        // Calculate y position with more chaotic effects
        const y = baseY - 
                 layerAmplitude * noiseVal + // Main wave pattern 
                 Math.sin(x * 0.06 * offset.freq + phase) * layerAmplitude * 0.6 + // INCREASED frequency and effect
                 Math.sin(x * 0.014 * offset.freq + phase * 0.8 + offset.phase) * layerAmplitude * 0.4 + // INCREASED effect
                 (1 - layerDepth) * 130; // INCREASED vertical spread between layers
        
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