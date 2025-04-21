/**
 * DOM.js - DOM manipulation utilities for robustness visualization
 * Provides helper functions for working with DOM elements and events
 */

/**
 * Creates an element with specified attributes and children
 * @param {string} tag - Element tag name
 * @param {Object} attributes - Element attributes
 * @param {Array|string|Node} children - Child elements, text or HTML
 * @returns {HTMLElement} - Created element
 */
function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);
    
    // Set attributes
    for (const [key, value] of Object.entries(attributes)) {
      if (key === 'className') {
        element.className = value;
      } else if (key === 'dataset') {
        for (const [dataKey, dataValue] of Object.entries(value)) {
          element.dataset[dataKey] = dataValue;
        }
      } else if (key === 'style' && typeof value === 'object') {
        Object.assign(element.style, value);
      } else if (key.startsWith('on') && typeof value === 'function') {
        element.addEventListener(key.substring(2).toLowerCase(), value);
      } else {
        element.setAttribute(key, value);
      }
    }
    
    // Add children
    if (children) {
      appendChildren(element, children);
    }
    
    return element;
  }
  
  /**
   * Appends children to an element
   * @param {HTMLElement} element - Parent element
   * @param {Array|string|Node} children - Child elements, text or HTML
   */
  function appendChildren(element, children) {
    if (!children) return;
    
    if (typeof children === 'string') {
      element.innerHTML = children;
    } else if (children instanceof Node) {
      element.appendChild(children);
    } else if (Array.isArray(children)) {
      for (const child of children) {
        if (child) {
          if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
          } else if (child instanceof Node) {
            element.appendChild(child);
          }
        }
      }
    }
  }
  
  /**
   * Removes all children from an element
   * @param {HTMLElement} element - Element to clear
   */
  function clearElement(element) {
    while (element.firstChild) {
      element.removeChild(element.firstChild);
    }
  }
  
  /**
   * Creates a document fragment with specified children
   * @param {Array|string|Node} children - Child elements, text or HTML
   * @returns {DocumentFragment} - Created fragment
   */
  function createFragment(children = []) {
    const fragment = document.createDocumentFragment();
    appendChildren(fragment, children);
    return fragment;
  }
  
  /**
   * Finds an element by selector with error handling
   * @param {string} selector - CSS selector
   * @param {HTMLElement} context - Context element
   * @returns {HTMLElement|null} - Found element or null
   */
  function findElement(selector, context = document) {
    try {
      return context.querySelector(selector);
    } catch (error) {
      console.error(`Error finding element with selector ${selector}:`, error);
      return null;
    }
  }
  
  /**
   * Finds all elements matching selector with error handling
   * @param {string} selector - CSS selector
   * @param {HTMLElement} context - Context element
   * @returns {Array} - Array of found elements
   */
  function findElements(selector, context = document) {
    try {
      return Array.from(context.querySelectorAll(selector));
    } catch (error) {
      console.error(`Error finding elements with selector ${selector}:`, error);
      return [];
    }
  }
  
  /**
   * Adds event listener with error handling
   * @param {HTMLElement} element - Element to attach listener to
   * @param {string} eventType - Event type
   * @param {Function} handler - Event handler
   * @param {Object} options - Event listener options
   */
  function addEvent(element, eventType, handler, options = {}) {
    try {
      element.addEventListener(eventType, handler, options);
    } catch (error) {
      console.error(`Error adding ${eventType} event:`, error);
    }
  }
  
  /**
   * Removes event listener with error handling
   * @param {HTMLElement} element - Element to remove listener from
   * @param {string} eventType - Event type
   * @param {Function} handler - Event handler
   * @param {Object} options - Event listener options
   */
  function removeEvent(element, eventType, handler, options = {}) {
    try {
      element.removeEventListener(eventType, handler, options);
    } catch (error) {
      console.error(`Error removing ${eventType} event:`, error);
    }
  }
  
  /**
   * Sets or gets data attribute on element
   * @param {HTMLElement} element - Target element
   * @param {string} key - Data attribute key
   * @param {*} [value] - Value to set (if omitted, gets the value)
   * @returns {*} - Data attribute value when getting
   */
  function data(element, key, value) {
    if (value === undefined) {
      return element.dataset[key];
    }
    element.dataset[key] = value;
  }
  
  /**
   * Creates and shows a tooltip near an element
   * @param {HTMLElement} element - Element to attach tooltip to
   * @param {string} content - Tooltip content
   * @param {Object} options - Tooltip options
   * @returns {HTMLElement} - Tooltip element
   */
  function showTooltip(element, content, options = {}) {
    const defaults = {
      position: 'top', // top, bottom, left, right
      className: 'tooltip',
      duration: 3000, // ms, 0 for persistent
      offset: 5
    };
    
    const settings = { ...defaults, ...options };
    const tooltip = createElement('div', { className: `${settings.className} ${settings.position}` }, content);
    
    document.body.appendChild(tooltip);
    
    // Position the tooltip
    const elementRect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    let top, left;
    
    switch (settings.position) {
      case 'top':
        top = elementRect.top - tooltipRect.height - settings.offset;
        left = elementRect.left + (elementRect.width / 2) - (tooltipRect.width / 2);
        break;
      case 'bottom':
        top = elementRect.bottom + settings.offset;
        left = elementRect.left + (elementRect.width / 2) - (tooltipRect.width / 2);
        break;
      case 'left':
        top = elementRect.top + (elementRect.height / 2) - (tooltipRect.height / 2);
        left = elementRect.left - tooltipRect.width - settings.offset;
        break;
      case 'right':
        top = elementRect.top + (elementRect.height / 2) - (tooltipRect.height / 2);
        left = elementRect.right + settings.offset;
        break;
    }
    
    // Adjust position to make sure tooltip is in viewport
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
    
    top = Math.max(0, Math.min(top + scrollTop, window.innerHeight + scrollTop - tooltipRect.height));
    left = Math.max(0, Math.min(left + scrollLeft, window.innerWidth + scrollLeft - tooltipRect.width));
    
    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
    
    // Set auto-remove timer if duration > 0
    if (settings.duration > 0) {
      setTimeout(() => {
        if (tooltip.parentNode) {
          tooltip.parentNode.removeChild(tooltip);
        }
      }, settings.duration);
    }
    
    return tooltip;
  }
  
  /**
   * Hides a tooltip
   * @param {HTMLElement} tooltip - Tooltip element to hide
   */
  function hideTooltip(tooltip) {
    if (tooltip && tooltip.parentNode) {
      tooltip.parentNode.removeChild(tooltip);
    }
  }
  
  /**
   * Checks if an element is visible in the viewport
   * @param {HTMLElement} element - Element to check
   * @param {number} threshold - Visibility threshold (0-1)
   * @returns {boolean} - True if element is visible
   */
  function isElementInViewport(element, threshold = 0) {
    const rect = element.getBoundingClientRect();
    
    // Element is completely visible
    if (threshold === 0) {
      return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
      );
    }
    
    // Element is partially visible based on threshold
    const visibleHeight = Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0);
    const visibleWidth = Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0);
    
    const visibleArea = visibleHeight * visibleWidth;
    const elementArea = rect.height * rect.width;
    
    return visibleArea / elementArea >= threshold;
  }
  
  /**
   * Handles table pagination
   * @param {HTMLElement} tableContainer - Container element
   * @param {HTMLElement} table - Table element
   * @param {Object} options - Pagination options
   */
  function setupTablePagination(tableContainer, table, options = {}) {
    const defaults = {
      rowsPerPage: 10,
      pageControlsSelector: '.pagination-controls',
      prevBtnSelector: '[id^="prev-page"]',
      nextBtnSelector: '[id^="next-page"]',
      infoSelector: '[id^="pagination-info"]',
      rowSelector: 'tbody tr'
    };
    
    const settings = { ...defaults, ...options };
    const rows = findElements(settings.rowSelector, table);
    const totalRows = rows.length;
    const totalPages = Math.ceil(totalRows / settings.rowsPerPage);
    let currentPage = 1;
    
    const controls = findElement(settings.pageControlsSelector, tableContainer);
    const prevBtn = findElement(settings.prevBtnSelector, controls);
    const nextBtn = findElement(settings.nextBtnSelector, controls);
    const infoSpan = findElement(settings.infoSelector, controls);
    
    function updatePageInfo() {
      if (infoSpan) {
        infoSpan.textContent = `Page ${currentPage} of ${totalPages}`;
      }
    }
    
    function updateButtonStates() {
      if (prevBtn) {
        prevBtn.disabled = currentPage === 1;
      }
      if (nextBtn) {
        nextBtn.disabled = currentPage === totalPages;
      }
    }
    
    function showPage(page) {
      const start = (page - 1) * settings.rowsPerPage;
      const end = start + settings.rowsPerPage;
      
      rows.forEach((row, index) => {
        row.style.display = (index >= start && index < end) ? '' : 'none';
      });
      
      currentPage = page;
      updatePageInfo();
      updateButtonStates();
    }
    
    // Initialize pagination
    showPage(1);
    
    // Add event listeners
    if (prevBtn) {
      addEvent(prevBtn, 'click', () => {
        if (currentPage > 1) {
          showPage(currentPage - 1);
        }
      });
    }
    
    if (nextBtn) {
      addEvent(nextBtn, 'click', () => {
        if (currentPage < totalPages) {
          showPage(currentPage + 1);
        }
      });
    }
    
    // Return controller for external use
    return {
      goToPage: showPage,
      getCurrentPage: () => currentPage,
      getTotalPages: () => totalPages,
      updateRowsPerPage: (rowsPerPage) => {
        settings.rowsPerPage = rowsPerPage;
        const newTotalPages = Math.ceil(totalRows / rowsPerPage);
        totalPages = newTotalPages;
        
        // Reset to page 1 if current page is now out of range
        if (currentPage > totalPages) {
          currentPage = 1;
        }
        
        showPage(currentPage);
      }
    };
  }
  
  /**
   * Sets up table sorting
   * @param {HTMLElement} table - Table element
   * @param {Object} options - Sorting options
   */
  function setupTableSorting(table, options = {}) {
    const defaults = {
      headerSelector: 'thead th.sortable',
      ascClass: 'sort-asc',
      descClass: 'sort-desc',
      activeClass: 'sort-active'
    };
    
    const settings = { ...defaults, ...options };
    const headers = findElements(settings.headerSelector, table);
    
    function sortTable(header, direction) {
      const sortKey = header.getAttribute('data-sort');
      if (!sortKey) return;
      
      const isAsc = direction === 'asc';
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      
      // Get column index
      const cellIndex = Array.from(header.parentNode.children).indexOf(header);
      
      // Sort rows
      rows.sort((a, b) => {
        const aValue = getCellValue(a, cellIndex, sortKey);
        const bValue = getCellValue(b, cellIndex, sortKey);
        
        if (typeof aValue === 'number' && typeof bValue === 'number') {
          return isAsc ? aValue - bValue : bValue - aValue;
        }
        
        return isAsc
          ? String(aValue).localeCompare(String(bValue))
          : String(bValue).localeCompare(String(aValue));
      });
      
      // Remove existing rows
      clearElement(tbody);
      
      // Append sorted rows
      const fragment = createFragment(rows);
      tbody.appendChild(fragment);
      
      // Update header classes
      headers.forEach(h => {
        h.classList.remove(settings.ascClass, settings.descClass, settings.activeClass);
      });
      
      header.classList.add(settings.activeClass);
      header.classList.add(isAsc ? settings.ascClass : settings.descClass);
    }
    
    function getCellValue(row, index, sortKey) {
      const cell = row.cells[index];
      
      // Try to get data attribute first
      if (sortKey && cell.dataset[sortKey] !== undefined) {
        const value = cell.dataset[sortKey];
        return isNaN(value) ? value : parseFloat(value);
      }
      
      // Get cell text content
      const text = cell.textContent.trim();
      
      // Try to parse as number
      return isNaN(text) ? text : parseFloat(text);
    }
    
    // Add click event to sortable headers
    headers.forEach(header => {
      addEvent(header, 'click', () => {
        const isAsc = !header.classList.contains(settings.ascClass);
        sortTable(header, isAsc ? 'asc' : 'desc');
      });
    });
    
    // Return sort controller
    return {
      sort: (sortKey, direction = 'asc') => {
        const header = headers.find(h => h.getAttribute('data-sort') === sortKey);
        if (header) {
          sortTable(header, direction);
        }
      }
    };
  }
  
  // Export the utilities
  export {
    createElement,
    appendChildren,
    clearElement,
    createFragment,
    findElement,
    findElements,
    addEvent,
    removeEvent,
    data,
    showTooltip,
    hideTooltip,
    isElementInViewport,
    setupTablePagination,
    setupTableSorting
  };