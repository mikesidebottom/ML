/**
 * Wave animation using p5.js
 * Based on original Processing code
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
    p.frameRate(60);
    
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
    for(let i = 0; i < 40; i++) {
      let x_wave = 0;
      
      // Use p5.js noise function for colors
      const brightness = 120 + p.noise(p.noise(i)*time*3) * 100;
      p.stroke(brightness);
      p.fill(brightness);
      
      while(x_wave < p.width) {
        // Calculate y position with noise
        const y_pos = 0.3 * p.height * p.noise(x_wave/205, time + i*0.05);
        
        // Draw points instead of lines for the complex pattern
        const pointSize = 1 + p.noise(i * 0.2) * 1.5;
        p.ellipse(x_wave, y_pos, pointSize, pointSize);
        
        x_wave += 2.5; // Small step size for detailed waves
      }
    }
    
    time += 0.0075;
  };
};

// Wait for DOM to load before creating the p5 instance
document.addEventListener('DOMContentLoaded', function() {
  // Create p5 instance with the sketch only if script.js hasn't initialized the canvas yet
  if (!document.getElementById('waveCanvas').getContext('2d')) {
    new p5(waveSketch);
  }
});