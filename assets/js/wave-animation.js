/**
 * Wave animation using p5.js
 * Creates a dots-based wave animation like the one in 0001.png
 */

// Variables for the sketch
let time = 0.0;
const bgColor = '#1F1F1F';

// Create a p5.js instance mode sketch to avoid conflicts with global variables
const waveSketch = (p) => {
  p.setup = function() {
    // Get container dimensions and create canvas to fit it
    const container = document.querySelector('.wave-animation-container');
    if (!container) return;
    
    const canvas = p.createCanvas(container.offsetWidth, container.offsetHeight);
    canvas.parent('waveCanvas'); // Connect to the #waveCanvas element
    
    p.background(bgColor);
    p.frameRate(30); // Reduced framerate for better performance
    
    // Add resize listener
    window.addEventListener('resize', () => {
      if (container) {
        p.resizeCanvas(container.offsetWidth, container.offsetHeight);
      }
    });
  };

  p.draw = function() {
    p.background(bgColor);
    
    // Position the waves in the bottom part of the screen
    p.translate(0, p.height * 0.6);
    
    // Draw multiple waves for complex pattern
    const dotSpacing = 5; // Increase spacing for better performance
    const waveCount = 30;  // Reduced number of waves for better performance
    
    for(let i = 0; i < waveCount; i++) {
      // Use p5.js noise function for colors
      let alpha = p.map(i, 0, waveCount, 255, 50); // Fade out distant waves
      const brightness = p.map(i, 0, waveCount, 255, 150); // Brighter in front
      
      p.noStroke();
      p.fill(brightness, brightness, brightness, alpha);
      
      for(let x_wave = 0; x_wave < p.width; x_wave += dotSpacing) {
        // Calculate y position with noise
        const noiseScale = 0.005;
        const waveHeight = p.map(i, 0, waveCount, 0.15, 0.05);
        const y_pos = p.height * waveHeight * p.noise(x_wave * noiseScale, time + i * 0.2);
        
        // Draw points with varying sizes
        const pointSize = p.map(i, 0, waveCount, 3, 1);
        p.ellipse(x_wave, y_pos, pointSize, pointSize);
      }
    }
    
    time += 0.01;
  };
};

// Wait for DOM to load before creating the p5 instance
document.addEventListener('DOMContentLoaded', function() {
  // Only create the canvas once
  new p5(waveSketch);
});