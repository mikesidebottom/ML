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
  
  // Mouse interaction variables
  let mouseX = 0;
  let mouseY = 0;
  let targetMouseX = 0;
  let targetMouseY = 0;
  const mouseDampingFactor = 0.05; // Lower = smoother, less reactive
  const mouseInfluence = 0.3; // How much the mouse affects the waves (0-1)
  
  // Track mouse position with damping for smoothness
  canvas.addEventListener('mousemove', function(e) {
    // Get mouse position relative to canvas
    const rect = canvas.getBoundingClientRect();
    targetMouseX = (e.clientX - rect.left) / canvas.width;
    targetMouseY = (e.clientY - rect.top) / canvas.height;
  });
  
  // Constants for animation - SIGNIFICANTLY REDUCED for better performance
  const bgColor = '#1F1F1F';
  const waveCount = 15; // FURTHER REDUCED from 25 to 15 for much better performance
  const dotSpacing = 8; // INCREASED spacing significantly to reduce total dots rendered
  const amplitude = 220;
  const baseWaveFreq = 0.018;
  const speed = 0.015; // REDUCED speed slightly to compensate for fewer frames
  
  // Simplified random offset with fewer waves
  const randomOffsets = Array.from({length: waveCount}, () => ({
    freq: (Math.random() * 0.025) + 0.01,
    phase: Math.random() * Math.PI * 4,
    amp: (Math.random() * 0.6) + 0.8,
    speed: (Math.random() * 0.7) + 0.6,
    initialX: Math.random() * 800,
    initialY: Math.random() * 800,
    turbulence: (Math.random() * 0.5) + 0.7
  }));
  
  // Simplified noise function with fewer computations
  function noise(x, y, i) {
    const offset = randomOffsets[i % randomOffsets.length];
    const waveFreq = baseWaveFreq * offset.freq;
    const wavePhase = offset.phase;
    const waveTime = time * offset.speed;
    
    const turbX = x + offset.initialX;
    const turbY = y + offset.initialY;
    
    // Add subtle mouse influence to the noise
    const mouseEffect = Math.sin((turbX * 0.01) + (mouseX * 5)) * 
                        Math.cos((turbY * 0.01) + (mouseY * 5)) * 
                        mouseInfluence;
    
    // Reduced complexity - fewer sine/cosine calculations
    return (
      Math.sin(turbX * waveFreq + waveTime + wavePhase) * 0.5 + 
      Math.cos(turbX * waveFreq * 0.5 + waveTime * 1.2) * 0.3 +
      Math.sin(turbX * 0.02 * offset.turbulence + turbY * 0.02) * 0.2 +
      mouseEffect // Add mouse influence
    ) * 0.5 + 0.5;
  }
  
  // Handle window resize
  window.addEventListener('resize', function() {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
  });
  
  // Animation function with throttling
  let lastFrameTime = 0;
  const minFrameDelay = 25; // Limit to ~40fps max for better performance
  
  function animate(currentTime) {
    // Throttle frame rate
    if (currentTime - lastFrameTime < minFrameDelay) {
      requestAnimationFrame(animate);
      return;
    }
    lastFrameTime = currentTime;
    
    // Smooth mouse movement with damping
    mouseX += (targetMouseX - mouseX) * mouseDampingFactor;
    mouseY += (targetMouseY - mouseY) * mouseDampingFactor;
    
    // Clear the canvas
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Position for waves
    const baseY = canvas.height * 0.33;
    
    // Draw multiple wave layers from back to front
    for (let i = 0; i < waveCount; i++) {
      const layerDepth = i / waveCount;
      const offset = randomOffsets[i % randomOffsets.length];
      
      // Increased amplitude for individual waves since there are fewer of them
      const layerAmplitude = amplitude * (0.4 + layerDepth * 0.8) * offset.amp;
      
      // Optimize color calculations
      const brightness = Math.floor(120 + layerDepth * 135);
      const opacity = 0.1 + layerDepth * 0.9;
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Larger dots with fewer waves for similar visual impact
      const dotSize = 1.0 + layerDepth * 3.2;
      const xOffset = i * 25;
      
      // Draw wave with increased step size for better performance
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        const phase = time * (1 + layerDepth * 0.8) * offset.speed;
        const noiseVal = noise(x, i * 10, i);
        
        // Simplified y calculation with fewer sine calculations
        const y = baseY - 
                 layerAmplitude * noiseVal + 
                 Math.sin(x * 0.05 * offset.freq + phase) * layerAmplitude * 0.5 +
                 (1 - layerDepth) * 120;
        
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
  
  // Start the animation with the timing parameter
  requestAnimationFrame(animate);
});