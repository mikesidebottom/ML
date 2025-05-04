/**
 * Table of Contents Generator
 * Automatically generates a TOC from page headings
 */
document.addEventListener('DOMContentLoaded', function() {
  // Initialize TOC only if we have headings
  const headings = findPageHeadings();
  if (headings.length === 0) return;
  
  const tocContainer = createTocContainer();
  const processedHeadings = assignIdsToHeadings(headings);
  const tocList = buildTocStructure(processedHeadings);
  
  tocContainer.appendChild(tocList);
  document.body.appendChild(tocContainer);
  
  setupScrollHighlighting(processedHeadings);
  setupScrollPadding();
  
  // Initial scroll event to highlight current section
  setTimeout(function() {
    window.dispatchEvent(new Event('scroll'));
  }, 100);
});

/**
 * Find all headings in the page content
 * @returns {Array} Array of heading elements
 */
function findPageHeadings() {
  return Array.from(document.querySelectorAll(
    'main h1, main h2, main h3, main h4, .notebook-content h1, .notebook-content h2, .notebook-content h3, .notebook-content h4'
  )).filter(heading => heading.textContent.trim());
}

/**
 * Create the TOC container element
 * @returns {HTMLElement} The TOC container div
 */
function createTocContainer() {
  const tocContainer = document.createElement('div');
  tocContainer.className = 'toc-container';
  tocContainer.innerHTML = '<h3>Table of Contents</h3>';
  return tocContainer;
}

/**
 * Assign IDs to headings if they don't have them already
 * @param {Array} headings - Array of heading elements
 * @returns {Array} Same array with IDs assigned
 */
function assignIdsToHeadings(headings) {
  headings.forEach(heading => {
    if (!heading.id) {
      heading.id = heading.textContent.trim().toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-');
    }
  });
  return headings;
}

/**
 * Build the nested TOC structure
 * @param {Array} headings - Array of heading elements with IDs
 * @returns {HTMLElement} The TOC list element
 */
function buildTocStructure(headings) {
  const tocList = document.createElement('ul');
  tocList.className = 'toc-list';
  
  // Track heading levels for nesting
  let currentLevel = parseInt(headings[0].tagName.charAt(1));
  let currentList = tocList;
  let listStack = [tocList];
  
  headings.forEach(heading => {
    const headingLevel = parseInt(heading.tagName.charAt(1));
    
    // Create a list item for this heading
    const listItem = document.createElement('li');
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    listItem.appendChild(link);
    
    // Handle nesting of lists based on heading level
    if (headingLevel > currentLevel) {
      // This heading is a subheading, create a new nested list
      const nestedList = document.createElement('ul');
      if (listStack[listStack.length - 1].lastChild) {
        listStack[listStack.length - 1].lastChild.appendChild(nestedList);
        listStack.push(nestedList);
        currentList = nestedList;
        currentLevel = headingLevel;
      }
    } else if (headingLevel < currentLevel) {
      // This heading is higher level than previous, go back up the stack
      while (headingLevel < currentLevel && listStack.length > 1) {
        listStack.pop();
        currentLevel--;
      }
      currentList = listStack[listStack.length - 1];
    }
    
    // Add the list item to the current list
    currentList.appendChild(listItem);
  });
  
  return tocList;
}

/**
 * Set up scroll highlighting for TOC items
 * @param {Array} headings - Array of heading elements with IDs
 */
function setupScrollHighlighting(headings) {
  window.addEventListener('scroll', function() {
    const scrollPosition = window.scrollY + 100; // Offset for better accuracy
    
    // Find the current active heading
    let current = '';
    
    // Search from bottom to top to find last heading above viewport
    for (let i = headings.length - 1; i >= 0; i--) {
      if (headings[i].offsetTop <= scrollPosition) {
        current = headings[i].id;
        break;
      }
    }
    
    // Update active state in TOC
    document.querySelectorAll('.toc-list a').forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/**
 * Set up scroll padding to account for the fixed header
 */
function setupScrollPadding() {
  // Add scroll-padding to the HTML element to account for the fixed header
  const headerHeight = document.querySelector('header').offsetHeight;
  document.documentElement.style.scrollPaddingTop = (headerHeight + 20) + 'px';
  
  // Add custom click handler for TOC links
  document.querySelectorAll('.toc-list a').forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);
      
      if (targetElement) {
        // Get the current position of the target element
        const headerHeight = document.querySelector('header').offsetHeight;
        const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
        
        // Scroll to the target with offset for header
        window.scrollTo({
          top: targetPosition - headerHeight - 20, // Extra padding for visual comfort
          behavior: 'smooth'
        });
      }
    });
  });
}