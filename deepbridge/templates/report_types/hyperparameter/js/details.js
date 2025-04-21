/**
 * JavaScript for rendering detailed hyperparameter results
 */

// Global variables for pagination
let currentPage = 1;
let itemsPerPage = 10;
let filteredTrials = [];

// Initialize details tables when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Listen for the renderCharts event
    document.addEventListener('renderCharts', renderDetailedResults);
});

/**
 * Render detailed hyperparameter results
 */
function renderDetailedResults() {
    if (!window.reportData) {
        console.error('Cannot render detailed results: reportData not available');
        return;
    }
    
    console.log('Rendering detailed hyperparameter results...');
    
    // Render metrics table
    renderMetricsTable();
    
    // Render trials table with pagination
    renderTrialsTable();
}

/**
 * Render the metrics table
 */
function renderMetricsTable() {
    const container = document.getElementById('metrics-table-container');
    if (!container) return;
    
    // Get metrics data from report data
    const metricsData = window.reportData.metrics || {};
    
    if (Object.keys(metricsData).length === 0) {
        container.innerHTML = '<div class="alert alert-info">No metrics data available.</div>';
        return;
    }
    
    // Create table
    const table = document.createElement('table');
    table.className = 'data-table metrics-table';
    
    // Create table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Add rows for each metric
    Object.entries(metricsData).forEach(([metric, data]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${metric}</td>
            <td>${typeof data.value === 'number' ? data.value.toFixed(4) : data.value}</td>
            <td>${data.description || ''}</td>
        `;
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.innerHTML = '';
    container.appendChild(table);
}

/**
 * Render the trials table with pagination
 */
function renderTrialsTable() {
    const container = document.getElementById('trials-container');
    if (!container) return;
    
    // Get trials data from report data
    const trialsData = window.reportData.trials || [];
    
    if (trialsData.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No trial data available.</div>';
        return;
    }
    
    // Store all trials for filtering
    filteredTrials = [...trialsData];
    
    // Clear container
    container.innerHTML = '';
    
    // Create filter controls
    createFilterControls(container, trialsData);
    
    // Create table container
    const tableContainer = document.createElement('div');
    tableContainer.className = 'table-container';
    container.appendChild(tableContainer);
    
    // Render the paginated table
    renderPaginatedTrialsTable(tableContainer);
    
    // Create pagination controls
    createPaginationControls(container);
}

/**
 * Create filter controls for the trials table
 */
function createFilterControls(container, trialsData) {
    // Create filter controls container
    const filterControls = document.createElement('div');
    filterControls.className = 'filter-controls';
    
    // Create sort control
    const sortControl = document.createElement('div');
    sortControl.className = 'filter-control';
    
    const sortLabel = document.createElement('label');
    sortLabel.textContent = 'Sort by:';
    sortControl.appendChild(sortLabel);
    
    const sortSelect = document.createElement('select');
    sortSelect.id = 'sort-select';
    
    // Add default sorting options
    sortSelect.innerHTML = `
        <option value="trial_number">Trial Number</option>
        <option value="score" selected>Score (High to Low)</option>
        <option value="score_asc">Score (Low to High)</option>
    `;
    
    // Add parameter sorting options if available
    if (trialsData.length > 0 && trialsData[0].parameters) {
        Object.keys(trialsData[0].parameters).forEach(param => {
            const option = document.createElement('option');
            option.value = `param_${param}`;
            option.textContent = `Parameter: ${param}`;
            sortSelect.appendChild(option);
        });
    }
    
    sortSelect.addEventListener('change', () => {
        sortTrials(sortSelect.value);
        currentPage = 1; // Reset to first page
        renderPaginatedTrialsTable(document.querySelector('.table-container'));
        updatePaginationControls();
    });
    
    sortControl.appendChild(sortSelect);
    filterControls.appendChild(sortControl);
    
    // Create items per page control
    const paginationControl = document.createElement('div');
    paginationControl.className = 'filter-control';
    
    const paginationLabel = document.createElement('label');
    paginationLabel.textContent = 'Items per page:';
    paginationControl.appendChild(paginationLabel);
    
    const paginationSelect = document.createElement('select');
    paginationSelect.id = 'pagination-select';
    paginationSelect.innerHTML = `
        <option value="10" selected>10</option>
        <option value="25">25</option>
        <option value="50">50</option>
        <option value="100">100</option>
    `;
    
    paginationSelect.addEventListener('change', () => {
        itemsPerPage = parseInt(paginationSelect.value);
        currentPage = 1; // Reset to first page
        renderPaginatedTrialsTable(document.querySelector('.table-container'));
        updatePaginationControls();
    });
    
    paginationControl.appendChild(paginationSelect);
    filterControls.appendChild(paginationControl);
    
    // Create search control if we have a large number of trials
    if (trialsData.length > 10) {
        const searchControl = document.createElement('div');
        searchControl.className = 'filter-control';
        
        const searchLabel = document.createElement('label');
        searchLabel.textContent = 'Search:';
        searchControl.appendChild(searchLabel);
        
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.id = 'search-input';
        searchInput.placeholder = 'Filter trials...';
        
        searchInput.addEventListener('input', () => {
            filterTrials(searchInput.value);
            currentPage = 1; // Reset to first page
            renderPaginatedTrialsTable(document.querySelector('.table-container'));
            updatePaginationControls();
        });
        
        searchControl.appendChild(searchInput);
        filterControls.appendChild(searchControl);
    }
    
    container.appendChild(filterControls);
}

/**
 * Sort trials based on selected sort option
 */
function sortTrials(sortOption) {
    filteredTrials.sort((a, b) => {
        if (sortOption === 'trial_number') {
            return a.trial_number - b.trial_number;
        } else if (sortOption === 'score') {
            return b.score - a.score;
        } else if (sortOption === 'score_asc') {
            return a.score - b.score;
        } else if (sortOption.startsWith('param_')) {
            const param = sortOption.substring(6);
            const aValue = a.parameters[param];
            const bValue = b.parameters[param];
            
            if (typeof aValue === 'number' && typeof bValue === 'number') {
                return aValue - bValue;
            } else {
                return String(aValue).localeCompare(String(bValue));
            }
        }
        return 0;
    });
}

/**
 * Filter trials based on search input
 */
function filterTrials(searchTerm) {
    if (!searchTerm) {
        filteredTrials = [...window.reportData.trials];
        return;
    }
    
    searchTerm = searchTerm.toLowerCase();
    
    filteredTrials = window.reportData.trials.filter(trial => {
        // Search in trial number and score
        if (trial.trial_number.toString().includes(searchTerm) || 
            trial.score.toString().includes(searchTerm)) {
            return true;
        }
        
        // Search in parameters
        for (const [key, value] of Object.entries(trial.parameters)) {
            if (key.toLowerCase().includes(searchTerm) || 
                value.toString().toLowerCase().includes(searchTerm)) {
                return true;
            }
        }
        
        return false;
    });
}

/**
 * Render trials table with pagination
 */
function renderPaginatedTrialsTable(container) {
    // Clear container
    container.innerHTML = '';
    
    // Handle empty filtered results
    if (filteredTrials.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No trials match your filter criteria.</div>';
        return;
    }
    
    // Calculate pagination
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, filteredTrials.length);
    const paginatedTrials = filteredTrials.slice(startIndex, endIndex);
    
    // Create table
    const table = document.createElement('table');
    table.className = 'data-table trials-table';
    
    // Get all parameter keys from the first trial
    const parameterKeys = Object.keys(paginatedTrials[0].parameters);
    
    // Create table header
    const thead = document.createElement('thead');
    let headerRow = '<tr><th>Trial</th><th>Score</th>';
    
    // Add parameter columns
    parameterKeys.forEach(key => {
        headerRow += `<th>${key}</th>`;
    });
    
    headerRow += '</tr>';
    thead.innerHTML = headerRow;
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Find best trial for highlighting
    const bestTrial = window.reportData.best_trial;
    
    // Add rows for each trial
    paginatedTrials.forEach(trial => {
        const row = document.createElement('tr');
        
        // Highlight best trial
        if (bestTrial && trial.trial_number === bestTrial.trial_number) {
            row.className = 'best-trial';
        }
        
        // Add trial number and score
        let rowHtml = `
            <td>${trial.trial_number}</td>
            <td>${trial.score.toFixed(4)}</td>
        `;
        
        // Add parameter values
        parameterKeys.forEach(key => {
            const value = trial.parameters[key];
            rowHtml += `<td>${typeof value === 'number' ? value.toFixed(6) : value}</td>`;
        });
        
        row.innerHTML = rowHtml;
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.appendChild(table);
}

/**
 * Create pagination controls
 */
function createPaginationControls(container) {
    // Create pagination container
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'pagination';
    paginationContainer.id = 'pagination-controls';
    
    // Update pagination controls
    updatePaginationControls();
    
    container.appendChild(paginationContainer);
}

/**
 * Update pagination controls based on current state
 */
function updatePaginationControls() {
    const paginationContainer = document.getElementById('pagination-controls');
    if (!paginationContainer) return;
    
    // Calculate total pages
    const totalPages = Math.ceil(filteredTrials.length / itemsPerPage);
    
    // Clear pagination container
    paginationContainer.innerHTML = '';
    
    // Do not show pagination if only one page
    if (totalPages <= 1) return;
    
    // Previous button
    const prevButton = document.createElement('button');
    prevButton.textContent = '«';
    prevButton.disabled = currentPage === 1;
    prevButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderPaginatedTrialsTable(document.querySelector('.table-container'));
            updatePaginationControls();
        }
    });
    paginationContainer.appendChild(prevButton);
    
    // Page buttons
    const maxButtons = 5;
    const startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    const endPage = Math.min(totalPages, startPage + maxButtons - 1);
    
    for (let i = startPage; i <= endPage; i++) {
        const pageButton = document.createElement('button');
        pageButton.textContent = i;
        pageButton.className = i === currentPage ? 'active' : '';
        pageButton.addEventListener('click', () => {
            currentPage = i;
            renderPaginatedTrialsTable(document.querySelector('.table-container'));
            updatePaginationControls();
        });
        paginationContainer.appendChild(pageButton);
    }
    
    // Next button
    const nextButton = document.createElement('button');
    nextButton.textContent = '»';
    nextButton.disabled = currentPage === totalPages;
    nextButton.addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            renderPaginatedTrialsTable(document.querySelector('.table-container'));
            updatePaginationControls();
        }
    });
    paginationContainer.appendChild(nextButton);
}