// Distillation Report Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize any interactive components
    console.log('Distillation report loaded');

    // Tab switching functionality
    const tabs = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();

            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('show', 'active'));

            // Add active class to clicked tab
            this.classList.add('active');

            // Show corresponding content
            const targetId = this.getAttribute('data-bs-target');
            const targetContent = document.querySelector(targetId);
            if (targetContent) {
                targetContent.classList.add('show', 'active');
            }
        });
    });

    // Initialize Plotly charts if present
    if (window.Plotly && window.chartData) {
        initializeCharts();
    }

    // Table sorting functionality
    const tables = document.querySelectorAll('.sortable-table');
    tables.forEach(table => {
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => sortTable(table, index));
        });
    });
});

// Initialize Plotly charts
function initializeCharts() {
    // Check for chart containers and data
    const chartContainers = document.querySelectorAll('[data-chart-type]');

    chartContainers.forEach(container => {
        const chartType = container.getAttribute('data-chart-type');
        const chartId = container.id;

        if (window.chartData && window.chartData[chartType]) {
            renderChart(chartId, window.chartData[chartType]);
        }
    });
}

// Render a Plotly chart
function renderChart(containerId, chartData) {
    const container = document.getElementById(containerId);
    if (!container || !chartData) return;

    try {
        Plotly.newPlot(containerId, chartData.data, chartData.layout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
    } catch (error) {
        console.error(`Error rendering chart ${containerId}:`, error);
    }
}

// Table sorting
function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    // Determine sort direction
    const currentOrder = table.getAttribute('data-order') || 'asc';
    const newOrder = currentOrder === 'asc' ? 'desc' : 'asc';
    table.setAttribute('data-order', newOrder);

    // Sort rows
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();

        // Try to parse as number
        const aNum = parseFloat(aValue);
        const bNum = parseFloat(bValue);

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return newOrder === 'asc' ? aNum - bNum : bNum - aNum;
        }

        // Sort as text
        return newOrder === 'asc'
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
    });

    // Reorder rows in table
    rows.forEach(row => tbody.appendChild(row));
}

// Export functionality
function exportTableToCSV(tableId, filename) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const csv = [];
    const rows = table.querySelectorAll('tr');

    rows.forEach(row => {
        const cols = row.querySelectorAll('td, th');
        const csvRow = [];
        cols.forEach(col => {
            csvRow.push('"' + col.textContent.replace(/"/g, '""') + '"');
        });
        csv.push(csvRow.join(','));
    });

    downloadCSV(csv.join('\n'), filename);
}

function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}