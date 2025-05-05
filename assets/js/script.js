/**
 * Main JavaScript functionality for the workshop site
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all UI functionality
    initNavigation();
    initCardEffects();
    initNotebookInteractions();
    initSmoothScrolling();
    initButtonEffects();
    initCodeBlockInteractions();
    initWaveAnimation(); // Initialize the wave animation
});

/**
 * Handles active navigation item highlighting
 */
function initNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath === linkPath || 
            (currentPath.includes(linkPath) && linkPath !== '/')) {
            link.classList.add('active');
        }
        
        // Add pulse effect on hover
        link.addEventListener('mouseenter', function() {
            this.classList.add('nav-pulse');
        });
        
        link.addEventListener('mouseleave', function() {
            this.classList.remove('nav-pulse');
        });
    });
}

/**
 * Sets up hover effects for cards
 */
function initCardEffects() {
    // Regular cards hover effect with enhanced animation
    applyHoverEffect(
        document.querySelectorAll('.card'),
        {
            enter: {
                transform: 'translateY(-8px)',
                transition: 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
                boxShadow: '0 15px 20px rgba(0, 0, 0, 0.3)',
                borderLeft: '5px solid var(--accent-color)'
            },
            leave: {
                transform: 'translateY(0)',
                boxShadow: '',
                borderLeft: '3px solid var(--primary-color)'
            }
        }
    );
    
    // Notebook cards hover effect with enhanced animation
    applyHoverEffect(
        document.querySelectorAll('.notebook-card'),
        {
            enter: {
                transform: 'translateY(-8px) scale(1.02)',
                transition: 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
                boxShadow: '0 15px 25px rgba(0, 0, 0, 0.3)',
                borderLeft: '5px solid var(--accent-color)',
                backgroundColor: 'var(--card-bg-color)'
            },
            leave: {
                transform: 'translateY(0) scale(1)',
                boxShadow: '',
                borderLeft: '',
                backgroundColor: ''
            }
        }
    );
}

/**
 * Sets up interactive effects for notebooks content
 */
function initNotebookInteractions() {
    // Add interactive effects to notebook container
    const notebookContainer = document.querySelector('.notebook-container');
    if (notebookContainer) {
        // Add pulsing highlight to code cells on page load
        setTimeout(function() {
            const codeCells = document.querySelectorAll('pre.code-terminal');
            codeCells.forEach((cell, index) => {
                setTimeout(() => {
                    cell.classList.add('pulse-highlight');
                    setTimeout(() => {
                        cell.classList.remove('pulse-highlight');
                    }, 1000);
                }, index * 300);
            });
        }, 1000);
        
        // Make headings interactive
        const headings = document.querySelectorAll('.notebook-content h1, .notebook-content h2, .notebook-content h3, .notebook-content h4');
        headings.forEach(heading => {
            // Add subtle indicator that heading is interactive
            const indicator = document.createElement('span');
            indicator.className = 'heading-indicator';
            indicator.innerHTML = '§';
            heading.appendChild(indicator);
            
            // Add click-to-copy ID functionality
            heading.style.cursor = 'pointer';
            heading.addEventListener('click', function() {
                const id = this.id;
                const url = window.location.href.split('#')[0] + '#' + id;
                navigator.clipboard.writeText(url).then(() => {
                    // Show feedback
                    const feedback = document.createElement('div');
                    feedback.className = 'copy-feedback';
                    feedback.textContent = 'Link copied to clipboard!';
                    document.body.appendChild(feedback);
                    
                    setTimeout(() => {
                        feedback.classList.add('show');
                        setTimeout(() => {
                            feedback.classList.remove('show');
                            setTimeout(() => {
                                document.body.removeChild(feedback);
                            }, 300);
                        }, 2000);
                    }, 10);
                });
            });
        });
        
        // Make images zoomable
        const images = document.querySelectorAll('.notebook-content img');
        images.forEach(img => {
            img.addEventListener('click', function() {
                if (this.classList.contains('zoomed')) {
                    this.classList.remove('zoomed');
                    this.style.cursor = 'zoom-in';
                } else {
                    this.classList.add('zoomed');
                    this.style.cursor = 'zoom-out';
                }
            });
            
            // Set initial cursor
            img.style.cursor = 'zoom-in';
        });
    }
}

/**
 * Add interactive effects to buttons
 */
function initButtonEffects() {
    const buttons = document.querySelectorAll('.colab-button, .button, button');
    buttons.forEach(button => {
        button.addEventListener('mousedown', function() {
            this.classList.add('button-active');
        });
        
        button.addEventListener('mouseup', function() {
            this.classList.remove('button-active');
        });
        
        button.addEventListener('mouseleave', function() {
            this.classList.remove('button-active');
        });
    });
}

/**
 * Add interactive effects to code blocks
 */
function initCodeBlockInteractions() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-code-button';
        copyButton.textContent = 'Copy';
        
        // Add button to code block's parent
        block.parentNode.classList.add('code-block-container');
        block.parentNode.appendChild(copyButton);
        
        // Add copy functionality
        copyButton.addEventListener('click', function() {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                this.textContent = 'Copied!';
                this.classList.add('copied');
                
                setTimeout(() => {
                    this.textContent = 'Copy';
                    this.classList.remove('copied');
                }, 2000);
            });
        });
    });
}

/**
 * Helper function to apply hover effects to elements
 * @param {NodeList} elements - Elements to apply hover effects to
 * @param {Object} styles - Styles to apply on enter and leave
 */
function applyHoverEffect(elements, styles) {
    elements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            Object.entries(styles.enter).forEach(([property, value]) => {
                this.style[property] = value;
            });
        });
        
        element.addEventListener('mouseleave', function() {
            Object.entries(styles.leave).forEach(([property, value]) => {
                this.style[property] = value;
            });
        });
    });
}

/**
 * Sets up smooth scrolling behavior for in-page links
 */
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetElement = document.querySelector(this.getAttribute('href'));
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Initializes the wave animation at the bottom of the page
 */
function initWaveAnimation() {
    const canvas = document.getElementById('waveCanvas');
    
    console.log('Wave animation initialization called');
    console.log('Canvas element found:', !!canvas);
    
    if (!canvas) return;
    
    try {
        const ctx = canvas.getContext('2d');
        let time = 0.0;
        
        // Get the site's background color from CSS variable
        const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-color').trim();
        const backgroundColor = bgColor || '#22201c'; // Use the CSS variable or fallback
        
        // Function to resize canvas to match container size
        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
            console.log('Canvas resized to:', canvas.width, 'x', canvas.height);
        }
        
        // Initial resize
        resizeCanvas();
        
        // Resize canvas when window size changes
        window.addEventListener('resize', resizeCanvas);
        
        // Create noise function (simplified Perlin noise)
        function noise(x, y = 0) {
            // Create a simple noise effect using sine waves with different frequencies
            const value = Math.sin(x * 0.1) * Math.cos(y * 0.1) * 
                         Math.sin((x + y) * 0.05) * 
                         Math.cos(Math.sqrt(x*x + y*y) * 0.05);
            return (value + 1) * 0.5; // Normalize to 0..1
        }
        
        // Animation loop
        function animate() {
            // Clear canvas with background color (match the site background)
            ctx.fillStyle = backgroundColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw waves predominantly at the bottom half of the page
            ctx.save();
            const bottomOffset = canvas.height * 0.6; // Position waves in bottom 40% of the canvas
            ctx.translate(0, bottomOffset);
            
            // Draw multiple wave layers with points instead of lines
            for(let i = 0; i < 40; i++) {
                let xWave = 0;
                
                // Calculate color based on noise and time
                const brightness = 120 + noise(noise(i) * time * 3) * 100;
                ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
                
                // Draw each wave with points for a more complex effect
                while(xWave < canvas.width) {
                    // Calculate y position using noise
                    const yPos = 0.3 * canvas.height * noise(xWave/205, time + i*0.05);
                    
                    // Draw points instead of lines
                    ctx.beginPath();
                    const pointSize = 1 + noise(i * 0.2) * 1.5;
                    ctx.arc(xWave, yPos, pointSize, 0, Math.PI * 2);
                    ctx.fill();
                    
                    xWave += 2.5; // Smaller step size for more detail
                }
            }
            
            // Reset transform
            ctx.restore();
            
            // Update time - slower for more subtle animation
            time += 0.0075;
            
            // Continue animation loop
            requestAnimationFrame(animate);
        }
        
        console.log('Starting wave animation as background effect');
        // Start animation
        animate();
        
    } catch (error) {
        console.error('Error initializing wave animation:', error);
    }
}