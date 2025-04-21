// DOM.js - placeholder
/**
 * DOM Utilities
 * 
 * Helper functions for DOM manipulation, event handling, and UI updates
 * specific to the resilience report.
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
      while (element.firstChild) {
        element.removeChild(element.firstChild);
      }
    }
    
    /**
     * Add/remove a class to/from an element
     * @param {HTMLElement} element - Target element
     * @param {string} className - Class to toggle
     * @param {boolean} [force] - If provided, add or remove based on this value
     */
    static toggleClass(element, className, force) {
      if (arguments.length === 3) {
        element.classList.toggle(className, force);
      } else {
        element.classList.toggle(className);
      }
    }
    
    /**
     * Find an element by selector
     * @param {string} selector - CSS selector
     * @param {HTMLElement} [context=document] - Search context
     * @return {HTMLElement} Found element or null
     */
    static findElement(selector, context = document) {
      return context.querySelector(selector);
    }
    
    /**
     * Find elements by selector
     * @param {string} selector - CSS selector
     * @param {HTMLElement} [context=document] - Search context
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
     * @param {Object} [options] - Event options
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
     * @param {Object} [options] - Event options
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
     * @param {Object} [options] - Tooltip options
     * @return {HTMLElement} Tooltip element
     */
    static showTooltip(target, content, options = {}) {
      const defaults = {
        position: 'top',  // top, bottom, left, right
        className: 'tooltip',
        duration: 3000  // ms, 0 for persistent
      };
      
      const settings = { ...defaults, ...options };
      
      // Create tooltip element
      const tooltip = DOM.createElement('div', {
        className: `${settings.className} ${settings.position}`,
        style: { position: 'absolute', zIndex: 1000 }
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
      
      if (left < 0) left = 0;
      if (left + tooltipRect.width > viewportWidth) left = viewportWidth - tooltipRect.width;
      if (top < 0) top = 0;
      if (top + tooltipRect.height > viewportHeight) top = viewportHeight - tooltipRect.height;
      
      // Apply position
      tooltip.style.top = `${top + window.scrollY}px`;
      tooltip.style.left = `${left + window.scrollX}px`;
      
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
     * Create a modal dialog
     * @param {Object} options - Modal options
     * @return {Object} Modal controller
     */
    static createModal(options = {}) {
      const defaults = {
        title: '',
        content: '',
        className: 'modal',
        width: 'auto',
        height: 'auto',
        buttons: [
          { text: 'Close', action: 'close', className: 'btn-secondary' }
        ],
        onClose: null
      };
      
      const settings = { ...defaults, ...options };
      
      // Create modal elements
      const overlay = DOM.createElement('div', {
        className: 'modal-overlay',
        style: {
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }
      });
      
      const modalElement = DOM.createElement('div', {
        className: settings.className,
        style: {
          width: settings.width,
          height: settings.height,
          backgroundColor: '#fff',
          borderRadius: '4px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
          display: 'flex',
          flexDirection: 'column',
          maxWidth: '90vw',
          maxHeight: '90vh'
        }
      });
      
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
      
      const title = DOM.createElement('h3', {
        className: 'modal-title',
        style: {
          margin: 0,
          fontSize: '18px'
        }
      }, settings.title);
      
      const closeBtn = DOM.createElement('button', {
        className: 'modal-close',
        style: {
          background: 'none',
          border: 'none',
          fontSize: '20px',
          cursor: 'pointer',
          padding: '0 5px'
        },
        onclick: () => close()
      }, '×');
      
      header.appendChild(title);
      header.appendChild(closeBtn);
      
      const body = DOM.createElement('div', {
        className: 'modal-body',
        style: {
          padding: '15px',
          overflowY: 'auto',
          flex: 1
        }
      }, settings.content);
      
      const footer = DOM.createElement('div', {
        className: 'modal-footer',
        style: {
          padding: '15px',
          borderTop: '1px solid #e0e0e0',
          display: 'flex',
          justifyContent: 'flex-end'
        }
      });
      
      // Add buttons
      settings.buttons.forEach(btn => {
        const button = DOM.createElement('button', {
          className: `btn ${btn.className || ''}`,
          style: {
            padding: '8px 12px',
            margin: '0 5px',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          },
          onclick: () => {
            if (btn.action === 'close') {
              close();
            } else if (typeof btn.action === 'function') {
              btn.action(modalController);
            }
          }
        }, btn.text);
        
        footer.appendChild(button);
      });
      
      modalElement.appendChild(header);
      modalElement.appendChild(body);
      modalElement.appendChild(footer);
      overlay.appendChild(modalElement);
      
      // Close function
      function close() {
        if (overlay.parentNode) {
          overlay.parentNode.removeChild(overlay);
          if (typeof settings.onClose === 'function') {
            settings.onClose();
          }
        }
      }
      
      // Modal controller
      const modalController = {
        element: modalElement,
        overlay: overlay,
        close: close,
        setTitle: (newTitle) => {
          title.textContent = newTitle;
        },
        setContent: (newContent) => {
          DOM.emptyElement(body);
          
          if (typeof newContent === 'string') {
            body.innerHTML = newContent;
          } else if (newContent instanceof Node) {
            body.appendChild(newContent);
          }
        }
      };
      
      // Show modal
      document.body.appendChild(overlay);
      
      return modalController;
    }
    
    /**
     * Create a notification
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
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            maxWidth: '350px',
            padding: '10px'
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
          animation: 'notification-slide-in 0.3s ease-out'
        }
      });
      
      // Style based on type
      switch (type) {
        case 'success':
          notification.style.backgroundColor = '#d4edda';
          notification.style.color = '#155724';
          notification.style.borderLeft = '4px solid #28a745';
          break;
        case 'error':
          notification.style.backgroundColor = '#f8d7da';
          notification.style.color = '#721c24';
          notification.style.borderLeft = '4px solid #dc3545';
          break;
        case 'warning':
          notification.style.backgroundColor = '#fff3cd';
          notification.style.color = '#856404';
          notification.style.borderLeft = '4px solid #ffc107';
          break;
        case 'info':
        default:
          notification.style.backgroundColor = '#d1ecf1';
          notification.style.color = '#0c5460';
          notification.style.borderLeft = '4px solid #17a2b8';
          break;
      }
      
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
            color: 'inherit',
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
      const defaults = {
        headerSelector: 'th[data-sort]',
        ascClass: 'sort-asc',
        descClass: 'sort-desc'
      };
      
      const settings = { ...defaults, ...options };
      const headers = DOM.findElements(settings.headerSelector, table);
      
      // Add click handlers to sortable headers
      headers.forEach(header => {
        header.style.cursor = 'pointer';
        
        DOM.addEvent(header, 'click', () => {
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
        });
      });
      
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
          
          // Compare
          if (valueA === valueB) return 0;
          if (isAsc) {
            return valueA < valueB ? -1 : 1;
          } else {
            return valueA < valueB ? 1 : -1;
          }
        });
        
        // Reorder rows
        DOM.emptyElement(tbody);
        tbody.append(...rows);
      }
      
      function getCellValue(row, sortKey) {
        const cell = row.querySelector(`[data-sort-value="${sortKey}"]`) || 
                    row.querySelector(`td:nth-child(${getColumnIndex(sortKey) + 1})`);
        
        if (!cell) return '';
        
        // Check if there's a specific sort value attribute
        if (cell.dataset.sortValue) {
          return cell.dataset.sortValue;
        }
        
        return cell.textContent.trim();
      }
      
      function getColumnIndex(sortKey) {
        for (let i = 0; i < headers.length; i++) {
          if (headers[i].dataset.sort === sortKey) {
            return i;
          }
        }
        return 0;
      }
      
      // Return controller for external use
      return {
        sort: (sortKey, isAsc = true) => {
          const header = headers.find(h => h.dataset.sort === sortKey);
          if (header) {
            // Clear sort classes from all headers
            headers.forEach(h => {
              h.classList.remove(settings.ascClass, settings.descClass);
            });
            
            // Add sort class to specified header
            header.classList.add(isAsc ? settings.ascClass : settings.descClass);
            
            // Sort the table
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
      const defaults = {
        caseSensitive: false,
        columns: null  // null for all columns
      };
      
      const settings = { ...defaults, ...options };
      
      // Add input handler
      DOM.addEvent(filterInput, 'input', () => {
        const filterValue = settings.caseSensitive ? 
          filterInput.value : filterInput.value.toLowerCase();
        
        filterTable(filterValue);
      });
      
      function filterTable(filterValue) {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        rows.forEach(row => {
          const cells = settings.columns ? 
            Array.from(row.querySelectorAll(`td:nth-child(${settings.columns.join('),td:nth-child(')}))`)) : 
            Array.from(row.querySelectorAll('td'));
          
          const rowText = cells.map(cell => {
            // Check for specific filter value attribute
            if (cell.dataset.filterValue) {
              return cell.dataset.filterValue;
            }
            return cell.textContent.trim();
          }).join(' ');
          
          const text = settings.caseSensitive ? rowText : rowText.toLowerCase();
          
          if (filterValue === '' || text.includes(filterValue)) {
            row.style.display = '';
          } else {
            row.style.display = 'none';
          }
        });
      }
      
      // Return controller for external use
      return {
        filter: (value) => {
          filterInput.value = value;
          const event = new Event('input');
          filterInput.dispatchEvent(event);
        },
        reset: () => {
          filterInput.value = '';
          const event = new Event('input');
          filterInput.dispatchEvent(event);
        }
      };
    }
    
    /**
     * Initialize tab navigation
     * @param {string} containerSelector - Tab container selector
     * @param {Object} options - Tab options
     * @return {Object} Tab controller
     */
    static initTabs(containerSelector, options = {}) {
      const defaults = {
        tabSelector: '.tab-button',
        contentSelector: '.tab-content',
        activeClass: 'active',
        defaultTab: 0
      };
      
      const settings = { ...defaults, ...options };
      const container = document.querySelector(containerSelector);
      
      if (!container) return null;
      
      const tabs = DOM.findElements(settings.tabSelector, container);
      const contents = DOM.findElements(settings.contentSelector, container);
      
      // Initialize with default tab
      const defaultTabIndex = typeof settings.defaultTab === 'number' ? 
        settings.defaultTab : 0;
      
      if (tabs.length > defaultTabIndex) {
        tabs[defaultTabIndex].classList.add(settings.activeClass);
        
        if (contents.length > defaultTabIndex) {
          contents[defaultTabIndex].classList.add(settings.activeClass);
        }
      }
      
      // Add click handlers to tabs
      tabs.forEach((tab, index) => {
        DOM.addEvent(tab, 'click', (e) => {
          e.preventDefault();
          
          // Deactivate all tabs and contents
          tabs.forEach(t => t.classList.remove(settings.activeClass));
          contents.forEach(c => c.classList.remove(settings.activeClass));
          
          // Activate clicked tab and corresponding content
          tab.classList.add(settings.activeClass);
          
          if (contents.length > index) {
            contents[index].classList.add(settings.activeClass);
          }
          
          // Trigger event
          const event = new CustomEvent('tab-changed', {
            detail: {
              container,
              tab,
              content: contents[index],
              index
            }
          });
          
          container.dispatchEvent(event);
        });
      });
      
      // Return controller for external use
      return {
        activate: (index) => {
          if (index >= 0 && index < tabs.length) {
            tabs[index].click();
          }
        },
        getActiveIndex: () => {
          return tabs.findIndex(tab => tab.classList.contains(settings.activeClass));
        }
      };
    }
  }
  
  export default DOM;