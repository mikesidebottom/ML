document.addEventListener('DOMContentLoaded', function() {
  // Always create a TOC container first
  const tocContainer = document.createElement('div');
  tocContainer.className = 'toc-container';
  tocContainer.innerHTML = '<h3>Table of Contents</h3>';
  
  // Find appropriate content container
  // Check all possible content containers on the page
  const contentContainers = [
    document.querySelector('.notebook-content'),
    document.querySelector('main.container'), 
    document.querySelector('.notebook-container')
  ];
  
  let contentContainer = null;
  let headings = [];
  
  // Try to find headings in any of the containers
  for (const container of contentContainers) {
    if (container) {
      const containerHeadings = Array.from(container.querySelectorAll('h1, h2, h3, h4'))
        .filter(heading => heading.textContent.trim());
      
      if (containerHeadings.length > 0) {
        contentContainer = container;
        headings = containerHeadings;
        break;
      }
    }
  }
  
  // Exit if no headings found
  if (!contentContainer || headings.length === 0) return;
  
  // Create the TOC list
  const tocList = document.createElement('ul');
  tocList.className = 'toc-list';
  
  // Process all headings and assign IDs if they don't have one
  headings.forEach(heading => {
    if (!heading.id) {
      heading.id = heading.textContent.trim().toLowerCase()
        .replace(/[^\w\s-]/g, '')    // Remove special chars
        .replace(/\s+/g, '-')        // Replace spaces with hyphens
        .replace(/-+/g, '-');        // Remove consecutive hyphens
    }
  });
  
  // Track heading levels to create nested lists
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
    
    // Handle nesting of lists
    if (headingLevel > currentLevel) {
      // Create a new nested list
      const nestedList = document.createElement('ul');
      if (listStack[listStack.length - 1].lastChild) {
        listStack[listStack.length - 1].lastChild.appendChild(nestedList);
        listStack.push(nestedList);
        currentList = nestedList;
        currentLevel = headingLevel;
      }
    } else if (headingLevel < currentLevel) {
      // Go back up the nesting level
      while (headingLevel < currentLevel && listStack.length > 1) {
        listStack.pop();
        currentLevel--;
      }
      currentList = listStack[listStack.length - 1];
    }
    
    // Add the list item to the current list
    currentList.appendChild(listItem);
  });
  
  // Add the TOC to the page
  tocContainer.appendChild(tocList);
  document.body.appendChild(tocContainer);
  
  // Add scroll highlighting for active section
  window.addEventListener('scroll', function() {
    const scrollPosition = window.scrollY;
    
    // Find the current active heading
    let current = '';
    
    headings.forEach(heading => {
      const sectionTop = heading.offsetTop - 100;
      if (scrollPosition >= sectionTop) {
        current = heading.id;
      }
    });
    
    // Highlight the current section in the TOC
    document.querySelectorAll('.toc-list a').forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
  
  // Style the toc list for better readability
  const styleElement = document.createElement('style');
  styleElement.textContent = `
    .toc-list {
      list-style-type: none;
      padding-left: 0;
      margin: 0;
      font-size: 0.85rem;
    }
    .toc-list ul {
      list-style-type: none;
      padding-left: 1rem;
      margin-top: 0.3rem;
      margin-bottom: 0.5rem;
    }
    .toc-list li {
      margin-bottom: 0.5rem;
      line-height: 1.3;
    }
    .toc-list a {
      color: var(--text-primary);
      text-decoration: none;
      display: inline-block;
      border-left: 2px solid transparent;
      padding-left: 0.5rem;
      transition: all 0.2s ease;
    }
    .toc-list a:hover {
      color: var(--primary-color);
      border-left-color: var(--primary-color);
    }
    .toc-list a.active {
      color: var(--primary-color);
      font-weight: 500;
      border-left-color: var(--primary-color);
    }
  `;
  document.head.appendChild(styleElement);
});