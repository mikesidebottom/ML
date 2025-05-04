/**
 * Main JavaScript functionality for the workshop site
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all UI functionality
    initNavigation();
    initCardEffects();
    initSmoothScrolling();
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
    });
}

/**
 * Sets up hover effects for cards
 */
function initCardEffects() {
    // Regular cards hover effect
    applyHoverEffect(
        document.querySelectorAll('.card'),
        {
            enter: {
                transform: 'translateY(-5px)',
                transition: 'all 0.3s ease'
            },
            leave: {
                transform: 'translateY(0)'
            }
        }
    );
    
    // Notebook cards hover effect
    applyHoverEffect(
        document.querySelectorAll('.notebook-card'),
        {
            enter: {
                transform: 'translateY(-3px)',
                transition: 'all 0.3s ease',
                borderLeft: '3px solid var(--accent-color)'
            },
            leave: {
                transform: 'translateY(0)',
                borderLeft: ''
            }
        }
    );
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