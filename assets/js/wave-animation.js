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
  
  // Mouse interaction variables - ENHANCED for more visible effect
  let mouseX = 0.5; // Default to center
  let mouseY = 0.5; 
  let targetMouseX = 0.5;
  let targetMouseY = 0.5;
  const mouseDampingFactor = 0.08; // Slightly more responsive
  const mouseInfluence = 0.6; // INCREASED from 0.3 for more noticeable effect
  
  // Track mouse position with improved calculation
  container.addEventListener('mousemove', function(e) {
    // Get mouse position relative to container (not canvas)
    const rect = container.getBoundingClientRect();
    targetMouseX = (e.clientX - rect.left) / rect.width;
    targetMouseY = (e.clientY - rect.top) / rect.height;
  });
  
  // Constants for animation - ADJUSTED to improve visual appeal while keeping performance
  const bgColor = '#1F1F1F';
  const waveCount = 18; // Slightly increased for better visuals
  const dotSpacing = 7; // Slightly reduced for more dots
  const amplitude = 230; // Slightly increased for more dramatic effect
  const baseWaveFreq = 0.018;
  const speed = 0.015;
  
  // Random offset with improved parameters for better visual effect
  const randomOffsets = Array.from({length: waveCount}, () => ({
    freq: (Math.random() * 0.03) + 0.01,
    phase: Math.random() * Math.PI * 5,
    amp: (Math.random() * 0.6) + 0.8,
    speed: (Math.random() * 0.7) + 0.6,
    initialX: Math.random() * 800,
    initialY: Math.random() * 800,
    turbulence: (Math.random() * 0.5) + 0.8 // Increased for more variation
  }));
  
  // Improved noise function with stronger mouse influence
  function noise(x, y, i) {
    const offset = randomOffsets[i % randomOffsets.length];
    const waveFreq = baseWaveFreq * offset.freq;
    const wavePhase = offset.phase;
    const waveTime = time * offset.speed;
    
    const turbX = x + offset.initialX;
    const turbY = y + offset.initialY;
    
    // Enhanced mouse effect - more directly impacts wave behavior
    // Mouse X affects horizontal wave frequency, Mouse Y affects amplitude
    const mouseXEffect = (mouseX - 0.5) * 2; // -1 to 1 range
    const mouseYEffect = (mouseY - 0.5) * 2; // -1 to 1 range
    
    // More pronounced and visible mouse influence
    const mouseEffect = 
      Math.sin((turbX * (0.01 + mouseYEffect * 0.01)) + (mouseXEffect * 6)) * 
      Math.cos((turbY * 0.01) + (mouseX * mouseY * 5)) * 
      mouseInfluence * (1 + mouseYEffect);
    
    // Improved sine wave combination for better visual effect
    return (
      Math.sin(turbX * waveFreq * (1 + mouseXEffect * 0.2) + waveTime + wavePhase) * 0.5 + 
      Math.cos(turbX * waveFreq * 0.6 + waveTime * 1.3) * 0.35 +
      Math.sin(turbX * 0.02 * offset.turbulence + turbY * 0.02) * 0.25 +
      mouseEffect
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
      
      // Enhanced dot appearance for better visual effect
      const brightness = Math.floor(120 + layerDepth * 135);
      const opacity = 0.12 + layerDepth * 0.88; // Slightly increased base opacity
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Adjusted dot sizes for better visual balance
      const dotSize = 0.9 + layerDepth * 3.0;
      const xOffset = i * 25;
      
      // Draw wave with improved y calculation
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        const phase = time * (1 + layerDepth * 0.8) * offset.speed;
        const noiseVal = noise(x, i * 10, i);
        
        // Enhanced y calculation with mouse influence
        const mouseYInfluence = (mouseY - 0.5) * 60 * layerDepth; // Higher layers move more with mouse
        const y = baseY - 
                 layerAmplitude * noiseVal + 
                 Math.sin(x * 0.05 * offset.freq + phase) * layerAmplitude * 0.5 +
                 (1 - layerDepth) * 120 + 
                 mouseYInfluence; // Direct mouse Y influence
        
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