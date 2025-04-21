/**
 * JavaScript for rendering hyperparameter importance visualizations
 */

// Initialize importance charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Listen for the renderCharts event
    document.addEventListener('renderCharts', renderImportanceVisualizations);
});

/**
 * Render importance visualizations
 */
function renderImportanceVisualizations() {
    if (!window.reportData || typeof Plotly === 'undefined') {
        console.error('Cannot render importance charts: reportData or Plotly not available');
        return;
    }

    console.log('Rendering hyperparameter importance visualizations...');
    
    // Render the importance bar chart
    renderImportanceChart();
    
    // Render the importance table
    renderImportanceTable();
}

/**
 * Render the hyperparameter importance bar chart
 */
function renderImportanceChart() {
    const container = document.getElementById('importance-chart-container');
    if (!container) return;
    
    // Clear any existing content
    container.innerHTML = '';
    
    // Create chart element
    const chartElement = document.createElement('div');
    chartElement.id = 'importance-chart';
    chartElement.style.height = '400px';
    container.appendChild(chartElement);
    
    // Get importance scores from report data
    const importanceData = window.reportData.importance_scores || [];
    
    if (importanceData.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No importance data available.</div>';
        return;
    }
    
    // Sort data by importance score in descending order
    const sortedData = [...importanceData].sort((a, b) => b.score - a.score);
    
    // Prepare data for Plotly
    const params = sortedData.map(item => item.parameter);
    const scores = sortedData.map(item => item.score);
    
    // Define color scale based on importance
    const colors = scores.map(score => {
        // Normalize score to [0, 1] for color scale
        const normalizedScore = score / Math.max(...scores);
        return `rgba(74, 111, 235, ${normalizedScore.toFixed(2)})`;
    });
    
    // Create the plot
    const data = [{
        type: 'bar',
        x: params,
        y: scores,
        marker: {
            color: colors
        }
    }];
    
    const layout = {
        title: 'Hyperparameter Importance',
        xaxis: {
            title: 'Hyperparameter',
            automargin: true
        },
        yaxis: {
            title: 'Importance Score'
        },
        margin: {
            l: 50,
            r: 50,
            b: 100,
            t: 50,
            pad: 4
        }
    };
    
    Plotly.newPlot('importance-chart', data, layout, {responsive: true});
}

/**
 * Render the hyperparameter importance table
 */
function renderImportanceTable() {
    const container = document.getElementById('importance-table-container');
    if (!container) return;
    
    // Get importance scores from report data
    const importanceData = window.reportData.importance_scores || [];
    
    if (importanceData.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No importance data available.</div>';
        return;
    }
    
    // Sort data by importance score in descending order
    const sortedData = [...importanceData].sort((a, b) => b.score - a.score);
    
    // Create table
    const table = document.createElement('table');
    table.className = 'data-table importance-table';
    
    // Create table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Rank</th>
            <th>Parameter</th>
            <th>Importance Score</th>
            <th>Description</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Add rows for each parameter
    sortedData.forEach((item, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${item.parameter}</td>
            <td>${item.score.toFixed(4)}</td>
            <td>${item.description || ''}</td>
        `;
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.innerHTML = '';
    container.appendChild(table);
}