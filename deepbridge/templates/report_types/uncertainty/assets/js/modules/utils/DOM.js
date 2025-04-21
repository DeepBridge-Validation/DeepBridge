/**
 * DOM Utilities
 * 
 * Helper functions for DOM manipulation, event handling, and UI updates
 * in the uncertainty report.
 */

class DOM {
    /**
     * Create a DOM element with attributes and children
     * @param {string} tag - Element tag name
     * @param {Object} attributes - Element attributes
     * @param {string|Array|Node} children - Element children
     * @return {HTMLElement} Created element
     */
    static createElement(tag, attributes = {}, children = null) {
      const element = document.createElement(tag);
      
      // Set attributes
      Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'className') {
          element.className = value;
        } else if (key === 'style' && typeof value === 'object') {
          Object.assign(element.style, value);
        } else if (key.startsWith('on') && typeof value === 'function') {
          const eventType = key.substring(2).toLowerCase();
          element.addEventListener(eventType, value);
        } else if (key === 'dataset' && typeof value === 'object') {
          Object.entries(value).forEach(([dataKey, dataValue]) => {
            element.dataset[dataKey] = dataValue;
          });
        } else {
          element.setAttribute(key, value);
        }
      });
      
      // Append children
      if (children !== null) {
        if (Array.isArray(children)) {
          children.forEach(child => {
            DOM.appendChildToElement(element, child);
          });
        } else {
          DOM.appendChildToElement(element, children);
        }
      }
      
      return element;
    }
    
    /**
     * Append a child to an element
     * @param {HTMLElement} element - Parent element
     * @param {string|Node} child - Child to append
     */
    static appendChildToElement(element, child) {
      if (typeof child === 'string') {
        element.appendChild(document.createTextNode(child));
      } else if (child instanceof Node) {
        element.appendChild(child);
      }
    }
    
    /**
     * Create a document fragment with children
     * @param {Array} children - Fragment children
     * @return {DocumentFragment} Created fragment
     */
    static createFragment(children = []) {
      const fragment = document.createDocumentFragment();
      
      children.forEach(child => {
        DOM.appendChildToElement(fragment, child);
      });
      
      return fragment;
    }
    
    /**
     * Empty an element (remove all children)
     * @param {HTMLElement} element - Element to empty
     */
    static emptyElement(element) {
      if (!element) return;
      
      while (element.firstChild) {
        element.removeChild(element.firstChild);
      }
    }
    
    /**
     * Find an element by selector
     * @param {string} selector - CSS selector
     * @param {HTMLElement} context - Search context (default: document)
     * @return {HTMLElement} Found element or null
     */
    static findElement(selector, context = document) {
      return context.querySelector(selector);
    }
    
    /**
     * Find elements by selector
     * @param {string} selector - CSS selector
     * @param {HTMLElement} context - Search context (default: document)
     * @return {Array} Array of found elements
     */
    static findElements(selector, context = document) {
      return Array.from(context.querySelectorAll(selector));
    }
    
    /**
     * Add event listener with error handling
     * @param {HTMLElement} element - Element to attach to
     * @param {string} type - Event type
     * @param {Function} listener - Event handler
     * @param {Object} options - Event options
     */
    static addEvent(element, type, listener, options) {
      try {
        element.addEventListener(type, listener, options);
      } catch (error) {
        console.error(`Error adding ${type} event:`, error);
      }
    }
    
    /**
     * Remove event listener with error handling
     * @param {HTMLElement} element - Element to detach from
     * @param {string} type - Event type
     * @param {Function} listener - Event handler
     * @param {Object} options - Event options
     */
    static removeEvent(element, type, listener, options) {
      try {
        element.removeEventListener(type, listener, options);
      } catch (error) {
        console.error(`Error removing ${type} event:`, error);
      }
    }
    
    /**
     * Create and show a tooltip
     * @param {HTMLElement} target - Target element
     * @param {string} content - Tooltip content
     * @param {Object} options - Tooltip options
     * @return {HTMLElement} Tooltip element
     */
    static showTooltip(target, content, options = {}) {
      const defaults = {
        position: 'top',  // top, bottom, left, right
        className: 'tooltip',
        duration: 3000    // ms, 0 for persistent
      };
      
      const settings = { ...defaults, ...options };
      
      // Remove any existing tooltips for this target
      const existingTooltip = document.querySelector(`.${settings.className}[data-for="${target.id}"]`);
      if (existingTooltip) {
        existingTooltip.remove();
      }
      
      // Create tooltip element
      const tooltip = DOM.createElement('div', {
        className: `${settings.className} ${settings.position}`,
        dataset: { for: target.id },
        style: {
          position: 'absolute',
          zIndex: 1000,
          padding: '6px 10px',
          borderRadius: '4px',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: '#fff',
          fontSize: '14px',
          pointerEvents: 'none',
          maxWidth: '300px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.2)'
        }
      }, content);
      
      document.body.appendChild(tooltip);
      
      // Position tooltip
      const targetRect = target.getBoundingClientRect();
      const tooltipRect = tooltip.getBoundingClientRect();
      
      let top, left;
      
      switch (settings.position) {
        case 'top':
          top = targetRect.top - tooltipRect.height - 8;
          left = targetRect.left + (targetRect.width / 2) - (tooltipRect.width / 2);
          break;
        case 'bottom':
          top = targetRect.bottom + 8;
          left = targetRect.left + (targetRect.width / 2) - (tooltipRect.width / 2);
          break;
        case 'left':
          top = targetRect.top + (targetRect.height / 2) - (tooltipRect.height / 2);
          left = targetRect.left - tooltipRect.width - 8;
          break;
        case 'right':
          top = targetRect.top + (targetRect.height / 2) - (tooltipRect.height / 2);
          left = targetRect.right + 8;
          break;
      }
      
      // Ensure tooltip stays in viewport
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      
      if (left < 10) left = 10;
      if (left + tooltipRect.width > viewportWidth - 10) left = viewportWidth - tooltipRect.width - 10;
      if (top < 10) top = 10;
      if (top + tooltipRect.height > viewportHeight - 10) top = viewportHeight - tooltipRect.height - 10;
      
      // Apply position
      tooltip.style.top = `${top + window.scrollY}px`;
      tooltip.style.left = `${left + window.scrollX}px`;
      
      // Add arrow element
      const arrow = DOM.createElement('div', {
        className: 'tooltip-arrow',
        style: {
          position: 'absolute',
          width: '0',
          height: '0',
          borderStyle: 'solid'
        }
      });
      
      // Position arrow
      switch (settings.position) {
        case 'top':
          arrow.style.borderWidth = '6px 6px 0 6px';
          arrow.style.borderColor = 'rgba(0, 0, 0, 0.8) transparent transparent transparent';
          arrow.style.bottom = '-6px';
          arrow.style.left = '50%';
          arrow.style.marginLeft = '-6px';
          break;
        case 'bottom':
          arrow.style.borderWidth = '0 6px 6px 6px';
          arrow.style.borderColor = 'transparent transparent rgba(0, 0, 0, 0.8) transparent';
          arrow.style.top = '-6px';
          arrow.style.left = '50%';
          arrow.style.marginLeft = '-6px';
          break;
        case 'left':
          arrow.style.borderWidth = '6px 0 6px 6px';
          arrow.style.borderColor = 'transparent transparent transparent rgba(0, 0, 0, 0.8)';
          arrow.style.right = '-6px';
          arrow.style.top = '50%';
          arrow.style.marginTop = '-6px';
          break;
        case 'right':
          arrow.style.borderWidth = '6px 6px 6px 0';
          arrow.style.borderColor = 'transparent rgba(0, 0, 0, 0.8) transparent transparent';
          arrow.style.left = '-6px';
          arrow.style.top = '50%';
          arrow.style.marginTop = '-6px';
          break;
      }
      
      tooltip.appendChild(arrow);
      
      // Set auto-removal
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
     * Hide a tooltip
     * @param {HTMLElement} tooltip - Tooltip element
     */
    static hideTooltip(tooltip) {
      if (tooltip && tooltip.parentNode) {
        tooltip.parentNode.removeChild(tooltip);
      }
    }
    
    /**
     * Create a notification toast
     * @param {string} message - Notification message
     * @param {string} type - Notification type (success, error, warning, info)
     * @param {Object} options - Notification options
     * @return {HTMLElement} Notification element
     */
    static showNotification(message, type = 'info', options = {}) {
      const defaults = {
        duration: 5000,  // ms, 0 for persistent
        position: 'top-right',  // top-right, top-left, bottom-right, bottom-left
        dismissible: true
      };
      
      const settings = { ...defaults, ...options };
      
      // Get or create container
      let container = document.querySelector(`.notification-container.${settings.position}`);
      
      if (!container) {
        container = DOM.createElement('div', {
          className: `notification-container ${settings.position}`,
          style: {
            position: 'fixed',
            zIndex: 1001,
            padding: '10px',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            maxWidth: '350px'
          }
        });
        
        // Position container
        if (settings.position.includes('top')) {
          container.style.top = '0';
        } else {
          container.style.bottom = '0';
        }
        
        if (settings.position.includes('right')) {
          container.style.right = '0';
        } else {
          container.style.left = '0';
        }
        
        document.body.appendChild(container);
      }
      
      // Determine notification color
      let bgColor, textColor, borderColor;
      
      switch (type) {
        case 'success':
          bgColor = '#d4edda';
          textColor = '#155724';
          borderColor = '#c3e6cb';
          break;
        case 'error':
          bgColor = '#f8d7da';
          textColor = '#721c24';
          borderColor = '#f5c6cb';
          break;
        case 'warning':
          bgColor = '#fff3cd';
          textColor = '#856404';
          borderColor = '#ffeeba';
          break;
        case 'info':
        default:
          bgColor = '#d1ecf1';
          textColor = '#0c5460';
          borderColor = '#bee5eb';
          break;
      }
      
      // Create notification element
      const notification = DOM.createElement('div', {
        className: `notification ${type}`,
        style: {
          padding: '12px 15px',
          borderRadius: '4px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: bgColor,
          color: textColor,
          borderLeft: `4px solid ${borderColor}`,
          animation: 'notification-slide-in 0.3s ease-out'
        },
        dataset: { type: type }
      });
      
      // Create message container
      const messageContainer = DOM.createElement('div', {
        className: 'notification-message'
      }, message);
      
      notification.appendChild(messageContainer);
      
      // Add dismiss button if dismissible
      if (settings.dismissible) {
        const dismissBtn = DOM.createElement('button', {
          className: 'notification-dismiss',
          style: {
            background: 'none',
            border: 'none',
            fontSize: '18px',
            cursor: 'pointer',
            marginLeft: '10px',
            padding: '0 5px',
            color: textColor,
            opacity: '0.5'
          },
          onclick: () => dismiss()
        }, '×');
        
        notification.appendChild(dismissBtn);
      }
      
      container.appendChild(notification);
      
      // Dismiss function
      function dismiss() {
        notification.style.animation = 'notification-slide-out 0.3s ease-out';
        
        setTimeout(() => {
          if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
            
            // Remove container if empty
            if (container.children.length === 0 && container.parentNode) {
              container.parentNode.removeChild(container);
            }
          }
        }, 300);
      }
      
      // Auto-dismiss
      if (settings.duration > 0) {
        setTimeout(dismiss, settings.duration);
      }
      
      return notification;
    }
    
    /**
     * Initialize table sorting functionality
     * @param {HTMLElement} table - Table element
     * @param {Object} options - Sorting options
     * @return {Object} Sorting controller
     */
    static initTableSort(table, options = {}) {
      if (!table) return null;
      
      const defaults = {
        headerSelector: 'th.sortable',
        ascClass: 'sort-asc',
        descClass: 'sort-desc'
      };
      
      const settings = { ...defaults, ...options };
      const headers = table.querySelectorAll(settings.headerSelector);
      
      // Add click handlers to sortable headers
      headers.forEach(header => {
        header.style.cursor = 'pointer';
        
        header.addEventListener('click', () => {
          const sortKey = header.dataset.sort;
          const isAsc = !header.classList.contains(settings.ascClass);
          
          // Clear sort classes from all headers
          headers.forEach(h => {
            h.classList.remove(settings.ascClass, settings.descClass);
          });
          
          // Add sort class to clicked header
          header.classList.add(isAsc ? settings.ascClass : settings.descClass);
          
          // Sort the table
          sortTable(table, sortKey, isAsc);
          
          // Dispatch sort event
          table.dispatchEvent(new CustomEvent('table-sorted', {
            detail: { sortKey, isAsc }
          }));
        });
      });
      
      /**
       * Sort table by specified column
       * @param {HTMLElement} table - Table to sort
       * @param {string} sortKey - Key to sort by
       * @param {boolean} isAsc - Sort ascending
       */
      function sortTable(table, sortKey, isAsc) {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        // Sort rows
        rows.sort((rowA, rowB) => {
          let valueA = getCellValue(rowA, sortKey);
          let valueB = getCellValue(rowB, sortKey);
          
          // Try to parse as numbers
          if (!isNaN(valueA) && !isNaN(valueB)) {
            valueA = parseFloat(valueA);
            valueB = parseFloat(valueB);
          }
          
          // Compare values
          if (valueA === valueB) return 0;
          
          const compareResult = valueA < valueB ? -1 : 1;
          return isAsc ? compareResult : -compareResult;
        });
        
        // Reorder rows
        DOM.emptyElement(tbody);
        rows.forEach(row => tbody.appendChild(row));
      }
      
      /**
       * Get cell value for sorting
       * @param {HTMLElement} row - Table row
       * @param {string} sortKey - Sort key
       * @return {string} Cell value
       */
      function getCellValue(row, sortKey) {
        // Try to find cell with sort-value data attribute first
        const cellWithValue = row.querySelector(`[data-sort-value="${sortKey}"]`);
        if (cellWithValue) {
          return cellWithValue.dataset.sortValue;
        }
        
        // Find cell by column index
        let columnIndex = 0;
        headers.forEach((header, index) => {
          if (header.dataset.sort === sortKey) {
            columnIndex = index;
          }
        });
        
        const cell = row.cells[columnIndex];
        return cell ? cell.textContent.trim() : '';
      }
      
      // Return controller for external use
      return {
        sort: (sortKey, isAsc = true) => {
          const header = Array.from(headers).find(h => h.dataset.sort === sortKey);
          if (header) {
            header.click();
          } else {
            // Sort directly if header not found
            sortTable(table, sortKey, isAsc);
          }
        }
      };
    }
    
    /**
     * Initialize table filtering functionality
     * @param {HTMLElement} table - Table element
     * @param {HTMLElement} filterInput - Filter input element
     * @param {Object} options - Filtering options
     * @return {Object} Filtering controller
     */
    static initTableFilter(table, filterInput, options = {}) {
      if (!table || !filterInput) return null;
      
      const defaults = {
        caseSensitive: false,
        columns: null  // null for all columns
      };
      
      const settings = { ...defaults, ...options };
      
      // Add input handler
      filterInput.addEventListener('input', () => {
        const filterValue = settings.caseSensitive ? 
          filterInput.value : filterInput.value.toLowerCase();
        
        filterTable(filterValue);
      });
      
      /**
       * Filter table rows
       * @param {string} filterValue - Value to filter by
       */
      function filterTable(filterValue) {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        // Filter rows
        rows.forEach(row => {
          let rowText = '';
          
          // Get text from specified columns or all columns
          if (settings.columns) {
            settings.columns.forEach(colIndex => {
              if (row.cells[colIndex]) {
                rowText += row.cells[colIndex].textContent.trim() + ' ';
              }
            });
          } else {
            // Get text from all columns
            Array.from(row.cells).forEach(cell => {
              rowText += cell.textContent.trim() + ' ';
            });
          }
          
          // Apply case sensitivity
          const searchableText = settings.caseSensitive ? rowText : rowText.toLowerCase();
          
          // Show/hide row
          row.style.display = filterValue === '' || searchableText.includes(filterValue) ? '' : 'none';
        });
        
        // Dispatch filter event
        table.dispatchEvent(new CustomEvent('table-filtered', {
          detail: { filterValue }
        }));
      }
      
      // Return controller for external use
      return {
        filter: (value) => {
          filterInput.value = value;
          
          // Trigger input event
          const event = new Event('input');
          filterInput.dispatchEvent(event);
        },
        reset: () => {
          filterInput.value = '';
          
          // Trigger input event
          const event = new Event('input');
          filterInput.dispatchEvent(event);
        }
      };
    }
    
    /**
     * Create a modal dialog
     * @param {Object} options - Modal options
     * @return {Object} Modal controller
     */
    static createModal(options = {}) {
      const defaults = {
        title: '',
        content: '',
        width: 'auto',
        maxWidth: '800px',
        buttons: [
          { text: 'Close', action: 'close', className: 'btn-secondary' }
        ],
        onClose: null,
        backdrop: true,
        draggable: false
      };
      
      const settings = { ...defaults, ...options };
      
      // Create backdrop
      const backdrop = DOM.createElement('div', {
        className: 'modal-backdrop',
        style: {
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          zIndex: 1000,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }
      });
      
      // Create modal
      const modal = DOM.createElement('div', {
        className: 'modal-dialog',
        style: {
          backgroundColor: '#fff',
          borderRadius: '4px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
          width: settings.width,
          maxWidth: settings.maxWidth,
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '90vh',
          position: 'relative'
        }
      });
      
      // Create modal header
      const header = DOM.createElement('div', {
        className: 'modal-header',
        style: {
          padding: '15px',
          borderBottom: '1px solid #e0e0e0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }
      });
      
      // Create title
      const titleElement = DOM.createElement('h3', {
        className: 'modal-title',
        style: {
          margin: 0,
          fontSize: '18px',
          fontWeight: 'bold'
        }
      }, settings.title);
      
      // Create close button
      const closeButton = DOM.createElement('button', {
        className: 'modal-close',
        style: {
          background: 'none',
          border: 'none',
          fontSize: '24px',
          cursor: 'pointer',
          color: '#999',
          padding: '0 5px'
        },
        onclick: () => close()
      }, '×');
      
      header.appendChild(titleElement);
      header.appendChild(closeButton);
      
      // Create modal body
      const body = DOM.createElement('div', {
        className: 'modal-body',
        style: {
          padding: '15px',
          overflowY: 'auto',
          flex: 1
        }
      });
      
      // Add content
      if (typeof settings.content === 'string') {
        body.innerHTML = settings.content;
      } else if (settings.content instanceof Node) {
        body.appendChild(settings.content);
      }
      
      // Create modal footer
      const footer = DOM.createElement('div', {
        className: 'modal-footer',
        style: {
          padding: '15px',
          borderTop: '1px solid #e0e0e0',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '10px'
        }
      });
      
      // Add buttons
      settings.buttons.forEach(button => {
        const btn = DOM.createElement('button', {
          className: `modal-btn ${button.className || ''}`,
          style: {
            padding: '8px 16px',
            borderRadius: '4px',
            cursor: 'pointer',
            border: 'none',
            backgroundColor: button.className?.includes('primary') ? '#1b78de' : '#e0e0e0',
            color: button.className?.includes('primary') ? '#fff' : '#333'
          },
          onclick: () => {
            if (button.action === 'close') {
              close();
            } else if (typeof button.action === 'function') {
              button.action(controller);
            }
          }
        }, button.text);
        
        footer.appendChild(btn);
      });
      
      // Assemble modal
      modal.appendChild(header);
      modal.appendChild(body);
      modal.appendChild(footer);
      backdrop.appendChild(modal);
      
      // Close function
      function close() {
        // Remove from DOM
        if (backdrop.parentNode) {
          backdrop.parentNode.removeChild(backdrop);
          
          // Call onClose callback
          if (typeof settings.onClose === 'function') {
            settings.onClose();
          }
        }
      }
      
      // Make draggable if enabled
      if (settings.draggable) {
        let isDragging = false;
        let offsetX, offsetY;
        
        header.style.cursor = 'move';
        
        header.addEventListener('mousedown', (e) => {
          isDragging = true;
          offsetX = e.clientX - modal.getBoundingClientRect().left;
          offsetY = e.clientY - modal.getBoundingClientRect().top;
          modal.style.position = 'absolute';
        });
        
        document.addEventListener('mousemove', (e) => {
          if (!isDragging) return;
          
          const x = e.clientX - offsetX;
          const y = e.clientY - offsetY;
          
          modal.style.left = `${x}px`;
          modal.style.top = `${y}px`;
        });
        
        document.addEventListener('mouseup', () => {
          isDragging = false;
        });
      }
      
      // Close when clicking backdrop if enabled
      if (settings.backdrop) {
        backdrop.addEventListener('click', (e) => {
          if (e.target === backdrop) {
            close();
          }
        });
      }
      
      // Add escape key handler
      document.addEventListener('keydown', function escHandler(e) {
        if (e.key === 'Escape') {
          close();
          document.removeEventListener('keydown', escHandler);
        }
      });
      
      // Modal controller
      const controller = {
        setTitle: (newTitle) => {
          titleElement.textContent = newTitle;
        },
        setContent: (newContent) => {
          DOM.emptyElement(body);
          
          if (typeof newContent === 'string') {
            body.innerHTML = newContent;
          } else if (newContent instanceof Node) {
            body.appendChild(newContent);
          }
        },
        close: close,
        getElement: () => modal
      };
      
      // Show modal
      document.body.appendChild(backdrop);
      
      return controller;
    }
  }
  
  export default DOM;