/**
 * Wave animation using p5.js
 * Creates a dots-based wave animation like the one in 0001.png
 */

// Create a p5.js instance mode sketch to avoid conflicts with global variables
const waveSketch = (p) => {
  // Variables for the sketch
  let time = 0.0;
  const bgColor = '#1F1F1F';
  
  p.setup = function() {
    console.log('Setting up wave animation');
    // Get container dimensions and create canvas to fit it
    const container = document.querySelector('.wave-animation-container');
    if (!container) {
      console.error('Wave animation container not found');
      return;
    }
    
    const canvas = p.createCanvas(container.offsetWidth, container.offsetHeight);
    canvas.parent('waveCanvas'); // Connect to the #waveCanvas element
    
    p.background(bgColor);
    p.frameRate(30); // Lower framerate for better performance
    
    console.log('Wave animation canvas created with size:', container.offsetWidth, 'x', container.offsetHeight);
    
    // Add resize listener
    window.addEventListener('resize', () => {
      if (container) {
        p.resizeCanvas(container.offsetWidth, container.offsetHeight);
        console.log('Wave animation canvas resized');
      }
    });
  };

  p.draw = function() {
    // Clear the background
    p.clear();
    p.background(bgColor);
    
    // Position the waves in the bottom part of the screen
    p.translate(0, p.height * 0.7);
    
    // Draw multiple waves for complex pattern
    const dotSpacing = 6; // Larger spacing for better performance
    const waveCount = 25;  // Reduced number of waves for better performance
    
    // Draw from back to front
    for(let i = 0; i < waveCount; i++) {
      // Calculate wave opacity and brightness
      const opacity = p.map(i, 0, waveCount, 50, 255); // More opaque in front
      const brightness = p.map(i, 0, waveCount, 150, 255); // Brighter in front
      
      p.noStroke();
      p.fill(brightness, brightness, brightness, opacity);
      
      // Draw each wave as a series of dots
      for(let x = 0; x < p.width; x += dotSpacing) {
        // Use noise for smooth wave movement
        const noiseScale = 0.003;
        const waveHeight = p.map(i, 0, waveCount, 0.05, 0.2);
        const y = -p.height * waveHeight * p.noise(x * noiseScale, time + i * 0.1);
        
        // Draw points with varying sizes
        const pointSize = p.map(i, 0, waveCount, 1, 3);
        p.ellipse(x, y, pointSize, pointSize);
      }
    }
    
    // Update time for animation
    time += 0.02;
  };
};

// Wait for DOM to load before creating the p5 instance
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, initializing wave animation');
  // Create p5 instance with the sketch
  new p5(waveSketch);
  
  // Check if canvas was created properly after a short delay
  setTimeout(() => {
    const canvas = document.querySelector('#waveCanvas > canvas');
    if (!canvas) {
      console.error('Wave animation canvas not found after initialization');
      // Try to recreate it
      new p5(waveSketch);
    } else {
      console.log('Wave animation is running with canvas dimensions:', canvas.width, 'x', canvas.height);
    }
  }, 1000);
});