/**
 * UI Widget utilities for DeepBridge reports
 */
const WidgetUtils = {
    /**
     * Initialize tab navigation
     * @param {string} tabsSelector - Selector for tab buttons container
     * @param {string} contentSelector - Selector for tab content containers
     * @param {Function} callback - Optional callback to run after tab change
     */
    initTabs: function(tabsSelector, contentSelector, callback) {
        const tabButtons = document.querySelectorAll(tabsSelector);
        if (!tabButtons.length) return;
        
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                tabButtons.forEach(btn => btn.classList.remove('active'));
                
                // Add active class to clicked button
                this.classList.add('active');
                
                // Show corresponding content
                const targetId = this.getAttribute('data-tab');
                const tabContents = document.querySelectorAll(contentSelector);
                
                tabContents.forEach(content => {
                    content.classList.remove('active');
                });
                
                const targetContent = document.getElementById(targetId);
                if (targetContent) {
                    targetContent.classList.add('active');
                    
                    // Run callback if provided
                    if (callback && typeof callback === 'function') {
                        callback(targetId, targetContent);
                    }
                }
            });
        });
        
        // Activate first tab by default if none is active
        if (!document.querySelector(`${tabsSelector}.active`)) {
            tabButtons[0]?.click();
        }
    },
    
    /**
     * Initialize chart selector
     * @param {string} selectorId - ID of the chart selector element
     * @param {string} containerSelector - Selector for chart containers
     */
    initChartSelector: function(selectorId, containerSelector) {
        const chartSelector = document.getElementById(selectorId);
        if (!chartSelector) return;
        
        const options = chartSelector.querySelectorAll('.chart-selector-option');
        options.forEach(option => {
            option.addEventListener('click', function() {
                // Remove active from all options
                options.forEach(opt => opt.classList.remove('active'));
                
                // Add active to clicked option
                this.classList.add('active');
                
                // Show corresponding chart
                const chartType = this.getAttribute('data-chart-type');
                const containers = document.querySelectorAll(containerSelector);
                
                containers.forEach(chart => {
                    chart.classList.remove('active');
                });
                
                const targetChart = document.querySelector(`${containerSelector}[data-chart-type="${chartType}"]`);
                if (targetChart) {
                    targetChart.classList.add('active');
                }
            });
        });
        
        // Activate first option by default if none is active
        if (!document.querySelector(`#${selectorId} .chart-selector-option.active`)) {
            options[0]?.click();
        }
    },
    
    /**
     * Initialize expandable sections
     * @param {string} headerSelector - Selector for section headers
     * @param {string} contentSelector - Selector for section content
     * @param {string} expandedClass - Class to apply when expanded
     */
    initExpandableSections: function(headerSelector, contentSelector, expandedClass = 'expanded') {
        const headers = document.querySelectorAll(headerSelector);
        
        headers.forEach(header => {
            header.addEventListener('click', function() {
                // Toggle expanded class on header
                this.classList.toggle(expandedClass);
                
                // Find the associated content
                const content = this.nextElementSibling;
                if (content && content.matches(contentSelector)) {
                    content.classList.toggle(expandedClass);
                }
                
                // Toggle icon if present
                const icon = this.querySelector('.panel-icon');
                if (icon) {
                    icon.classList.toggle(expandedClass);
                }
            });
        });
    },
    
    /**
     * Create a metric card
     * @param {string} label - Metric label
     * @param {string|number} value - Metric value
     * @param {string} description - Metric description
     * @param {string} colorClass - Color class for styling
     * @returns {HTMLElement} Metric card element
     */
    createMetricCard: function(label, value, description, colorClass = '') {
        const card = document.createElement('div');
        card.className = 'metric-card';
        
        const labelEl = document.createElement('div');
        labelEl.className = 'metric-card-label';
        labelEl.textContent = label;
        
        const valueEl = document.createElement('div');
        valueEl.className = `metric-card-value ${colorClass}`;
        valueEl.textContent = value;
        
        const descEl = document.createElement('div');
        descEl.className = 'metric-card-desc';
        descEl.textContent = description;
        
        card.appendChild(labelEl);
        card.appendChild(valueEl);
        card.appendChild(descEl);
        
        return card;
    },
    
    /**
     * Create a section with title and content
     * @param {string} title - Section title
     * @param {string} content - Section HTML content
     * @param {string} className - Additional CSS class
     * @returns {HTMLElement} Section element
     */
    createSection: function(title, content, className = '') {
        const section = document.createElement('div');
        section.className = `section ${className}`;
        
        const titleEl = document.createElement('h2');
        titleEl.className = 'section-title';
        titleEl.textContent = title;
        
        const contentEl = document.createElement('div');
        contentEl.className = 'section-content';
        contentEl.innerHTML = content;
        
        section.appendChild(titleEl);
        section.appendChild(contentEl);
        
        return section;
    },
    
    /**
     * Create a data table
     * @param {Array} headers - Table headers
     * @param {Array} rows - Table rows data
     * @param {string} className - Additional CSS class
     * @returns {HTMLElement} Table element
     */
    createTable: function(headers, rows, className = '') {
        const table = document.createElement('table');
        table.className = `data-table ${className}`;
        
        // Create header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        
        rows.forEach(row => {
            const tr = document.createElement('tr');
            
            row.forEach(cell => {
                const td = document.createElement('td');
                
                if (typeof cell === 'object' && cell !== null) {
                    // Cell with custom formatting
                    td.innerHTML = cell.html || '';
                    if (cell.className) td.className = cell.className;
                } else {
                    // Simple cell
                    td.textContent = cell;
                }
                
                tr.appendChild(td);
            });
            
            tbody.appendChild(tr);
        });
        
        table.appendChild(tbody);
        return table;
    },
    
    /**
     * Show a message box
     * @param {string} message - Message text
     * @param {string} type - Message type (info, warning, error, success)
     * @param {string} containerId - ID of container to show message in
     */
    showMessage: function(message, type = 'info', containerId) {
        const container = containerId ? document.getElementById(containerId) : document.body;
        if (!container) return;
        
        const msgBox = document.createElement('div');
        let icon = '📝';
        
        switch (type) {
            case 'warning':
                icon = '⚠️';
                msgBox.className = 'message-box warning';
                break;
            case 'error':
                icon = '❌';
                msgBox.className = 'message-box error';
                break;
            case 'success':
                icon = '✅';
                msgBox.className = 'message-box success';
                break;
            default:
                msgBox.className = 'message-box info';
        }
        
        msgBox.innerHTML = `
            <div class="message-content">
                <span class="message-icon">${icon}</span>
                <span class="message-text">${message}</span>
                <span class="message-close">&times;</span>
            </div>
        `;
        
        // Add close functionality
        const closeBtn = msgBox.querySelector('.message-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                msgBox.remove();
            });
        }
        
        // Add to container
        container.appendChild(msgBox);
        
        // Auto-remove after 5 seconds for non-error messages
        if (type !== 'error') {
            setTimeout(() => {
                msgBox.remove();
            }, 5000);
        }
        
        return msgBox;
    }
};