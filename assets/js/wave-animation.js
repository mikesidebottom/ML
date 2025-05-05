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
    const canvas = p.createCanvas(container.offsetWidth, container.offsetHeight);
    canvas.parent('waveCanvas'); // Connect to the #waveCanvas element
    
    p.background(bgColor);
    p.noStroke();
    p.smooth();
    p.frameRate(60);
    
    // Add resize listener
    window.addEventListener('resize', () => {
      p.resizeCanvas(container.offsetWidth, container.offsetHeight);
    });
  };

  p.draw = function() {
    p.background(bgColor);
    
    // Match the original processing code exactly from here:
    p.translate(0, p.height/3);
    p.stroke(220);
    
    for(let i = 0; i < 15*3.5; i++) {
      let x_wave = 0;
      
      // Use p5.js noise function
      p.stroke(255, p.noise(p.noise(i)*time*5)*220);
      
      while(x_wave < p.width) {
        // Draw points exactly as in Processing code
        p.point(x_wave, 0.5 * p.height * p.noise(x_wave/205, time + i*0.05));
        x_wave += 2.5;
      }
    }
    
    time += 0.0075;
  };
};

// Wait for DOM to load before creating the p5 instance
document.addEventListener('DOMContentLoaded', function() {
  // Create p5 instance with the sketch
  new p5(waveSketch);
});