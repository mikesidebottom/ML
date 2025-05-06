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
  
  // Mouse interaction variables - SIGNIFICANTLY ENHANCED
  let mouseX = 0.5; // Default to center
  let mouseY = 0.5; 
  let targetMouseX = 0.5;
  let targetMouseY = 0.5;
  const mouseDampingFactor = 0.1; // Increased for more responsive interaction
  const mouseInfluence = 0.85; // Greatly increased for much more noticeable effect
  
  // Track mouse position with improved calculation
  container.addEventListener('mousemove', function(e) {
    // Get mouse position relative to container (not canvas)
    const rect = container.getBoundingClientRect();
    targetMouseX = (e.clientX - rect.left) / rect.width;
    targetMouseY = (e.clientY - rect.top) / rect.height;
  });
  
  // Constants for animation - MORE WAVES while balancing performance
  const bgColor = '#1F1F1F';
  const waveCount = 28; // Increased from 18 to 28 for more layers
  const dotSpacing = 8; // Balance between performance and density
  const amplitude = 230;
  const baseWaveFreq = 0.018;
  const speed = 0.015;
  const horizontalMovementSpeed = 0.4; // Controls horizontal wave movement speed
  
  // Random offset with improved parameters for better visual effect
  const randomOffsets = Array.from({length: waveCount}, () => ({
    freq: (Math.random() * 0.03) + 0.01,
    phase: Math.random() * Math.PI * 5,
    amp: (Math.random() * 0.6) + 0.8,
    speed: (Math.random() * 0.7) + 0.6,
    initialX: Math.random() * 800,
    initialY: Math.random() * 800,
    turbulence: (Math.random() * 0.5) + 0.8,
    freqMultiplier: (Math.random() * 0.5) + 0.75, // New property for frequency variation
    horizontalOffset: Math.random() * 1000, // New property for horizontal movement
    horizontalSpeed: (Math.random() * 0.5 + 0.5) // Variable horizontal speeds
  }));
  
  // Improved noise function with MUCH stronger mouse influence and horizontal movement
  function noise(x, y, i, horizontalShift) {
    const offset = randomOffsets[i % randomOffsets.length];
    const waveFreq = baseWaveFreq * offset.freq;
    const wavePhase = offset.phase;
    const waveTime = time * offset.speed;
    
    // Apply horizontal movement
    const shiftedX = x + horizontalShift;
    
    const turbX = shiftedX + offset.initialX;
    const turbY = y + offset.initialY;
    
    // Enhanced mouse effect with dramatic frequency changes
    const mouseXEffect = (mouseX - 0.5) * 4; // -2 to 2 range (doubled impact)
    const mouseYEffect = (mouseY - 0.5) * 4; // -2 to 2 range (doubled impact)
    
    // Dynamic frequency that changes with mouse position
    const dynamicFreq = waveFreq * (1 + mouseXEffect * offset.freqMultiplier * 0.5);
    
    // Much more pronounced mouse influence
    const mouseEffect = 
      Math.sin((turbX * (0.015 + mouseYEffect * 0.02)) + (mouseXEffect * 8)) * 
      Math.cos((turbY * 0.015) + (mouseX * mouseY * 8)) * 
      mouseInfluence * (1 + Math.abs(mouseYEffect));
    
    // Wave calculation with dramatic mouse influence on frequency and horizontal movement
    return (
      Math.sin(turbX * dynamicFreq + waveTime + wavePhase + mouseXEffect) * 0.5 + 
      Math.cos(turbX * dynamicFreq * 0.6 + waveTime * 1.3 + mouseYEffect) * 0.35 +
      Math.sin(turbX * 0.02 * offset.turbulence * (1 + Math.abs(mouseYEffect) * 0.3) + 
              turbY * 0.02) * 0.25 +
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
      
      // Calculate horizontal movement for this layer
      // Different layers move at different speeds
      const horizontalShift = time * horizontalMovementSpeed * offset.horizontalSpeed * 100;
      
      // Calculate mouse-influenced horizontal shift
      // Mouse X position affects horizontal movement direction and speed
      const mouseHorizontalEffect = (mouseX - 0.5) * 200 * layerDepth;
      const totalHorizontalShift = horizontalShift + mouseHorizontalEffect;
      
      // Dynamic amplitude based on mouse Y position
      const mouseAmplitudeEffect = 1 + (mouseY - 0.5) * 0.6; // 0.7 to 1.3 range
      const layerAmplitude = amplitude * (0.4 + layerDepth * 0.8) * offset.amp * mouseAmplitudeEffect;
      
      // Enhanced dot appearance with mouse influence on brightness
      const mouseColorEffect = Math.abs(mouseX - 0.5) * 30; // 0 to 15 range
      const brightness = Math.floor(120 + layerDepth * 135 + mouseColorEffect);
      const opacity = 0.12 + layerDepth * 0.88;
      ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${opacity})`;
      
      // Dot sizes slightly affected by mouse X position
      const dotSizeEffect = 1 + (mouseX - 0.5) * 0.3; // 0.85 to 1.15 range
      const dotSize = (0.9 + layerDepth * 3.0) * dotSizeEffect;
      const xOffset = i * 25;
      
      // Draw wave with dramatically improved mouse influence and horizontal movement
      for (let x = -xOffset % dotSpacing; x < canvas.width; x += dotSpacing) {
        const phase = time * (1 + layerDepth * 0.8) * offset.speed;
        // Pass horizontal shift to noise function
        const noiseVal = noise(x, i * 10, i, totalHorizontalShift);
        
        // Enhanced y calculation with stronger mouse influence
        const mouseYInfluence = (mouseY - 0.5) * 100 * layerDepth; 
        const mouseXWaveEffect = Math.sin(x * 0.01 + mouseX * 10 + horizontalShift * 0.01) * 
                                20 * layerDepth * Math.abs(mouseX - 0.5);
        
        const y = baseY - 
                 layerAmplitude * noiseVal + 
                 Math.sin((x + totalHorizontalShift * 0.3) * 0.05 * offset.freq * 
                    (1 + Math.abs(mouseX - 0.5)) + phase) * layerAmplitude * 0.5 +
                 (1 - layerDepth) * 120 + 
                 mouseYInfluence + 
                 mouseXWaveEffect;
        
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