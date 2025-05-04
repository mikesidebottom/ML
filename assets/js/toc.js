document.addEventListener('DOMContentLoaded', function() {
  // Force create TOC for all pages
  const tocContainer = document.createElement('div');
  tocContainer.className = 'toc-container';
  tocContainer.innerHTML = '<h3>Table of Contents</h3>';
  
  // Find all headings in the document body
  const headings = Array.from(document.querySelectorAll('main h1, main h2, main h3, main h4, .notebook-content h1, .notebook-content h2, .notebook-content h3, .notebook-content h4'))
    .filter(heading => heading.textContent.trim());
  
  // Only continue if we found headings
  if (headings.length === 0) return;
  
  // Create the TOC list
  const tocList = document.createElement('ul');
  tocList.className = 'toc-list';
  
  // Assign IDs to headings if they don't have them
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
      const nestedList = document.createElement('ul');
      if (listStack[listStack.length - 1].lastChild) {
        listStack[listStack.length - 1].lastChild.appendChild(nestedList);
        listStack.push(nestedList);
        currentList = nestedList;
        currentLevel = headingLevel;
      }
    } else if (headingLevel < currentLevel) {
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
  
  // Add scroll highlighting
  window.addEventListener('scroll', function() {
    const scrollPosition = window.scrollY + 100;  // Offset to make active state more accurate
    
    // Find the current active heading
    let current = '';
    
    // Search in reverse order (last heading that's above viewport top)
    for (let i = headings.length - 1; i >= 0; i--) {
      if (headings[i].offsetTop <= scrollPosition) {
        current = headings[i].id;
        break;
      }
    }
    
    // Highlight the current section in the TOC
    document.querySelectorAll('.toc-list a').forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
  
  // Initial scroll event to highlight current section
  setTimeout(function() {
    window.dispatchEvent(new Event('scroll'));
  }, 100);
});