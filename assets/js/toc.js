document.addEventListener('DOMContentLoaded', function() {
  // Create TOC container
  const tocContainer = document.createElement('div');
  tocContainer.className = 'toc-container';
  tocContainer.innerHTML = '<h3>Table of Contents</h3>';
  
  // Find any content container with headers
  let contentContainer = null;
  let headings = [];
  
  // Try main content area first
  const mainContent = document.querySelector('main');
  if (mainContent) {
    // Find all headings in the main content
    headings = Array.from(mainContent.querySelectorAll('h1, h2, h3, h4'))
      .filter(heading => heading.textContent.trim());
    
    if (headings.length > 0) {
      contentContainer = mainContent;
    }
  }
  
  // If no headings found, try notebook content
  if (headings.length === 0) {
    const notebookContent = document.querySelector('.notebook-content');
    if (notebookContent) {
      headings = Array.from(notebookContent.querySelectorAll('h1, h2, h3, h4'))
        .filter(heading => heading.textContent.trim());
      
      if (headings.length > 0) {
        contentContainer = notebookContent;
      }
    }
  }
  
  // Exit if no headings found
  if (headings.length === 0) return;
  
  // Create the TOC list
  const tocList = document.createElement('ul');
  tocList.className = 'toc-list';
  
  // Process all headings and assign IDs
  headings.forEach(heading => {
    if (!heading.id) {
      heading.id = heading.textContent.trim().toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-');
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
});