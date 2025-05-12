// BoxplotController.js
const BoxplotController = {
    init: function() {
        console.log("Boxplot section initialized");
        
        // Initialize dropdown selector if it exists
        this.initModelSelector();
        
        // Initialize chart
        this.initCharts();
        
        // Initialize boxplot table
        this.fillBoxplotTable();
    },
    
    initModelSelector: function() {
        const modelSelector = document.getElementById('boxplot_model_selector');
        if (!modelSelector) return;
        
        modelSelector.addEventListener('change', (e) => {
            const selectedModel = e.target.value;
            this.updateBoxplotTable(selectedModel);
        });
    },
    
    initCharts: function() {
        console.log("Initializing boxplot charts");
        
        // Try to initialize all charts with a short delay to ensure DOM is ready
        setTimeout(() => {
            // Check if Plotly is available or can be loaded
            if (typeof Plotly !== 'undefined' || document.querySelector('script[src*="plotly"]')) {
                this.initializeBoxplotChart();
            } else {
                // Try to load Plotly from CDN
                const script = document.createElement('script');
                script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
                script.onload = () => this.initializeBoxplotChart();
                script.onerror = () => this.showChartError();
                document.head.appendChild(script);
            }
        }, 500);
    },
    
    initializeBoxplotChart: function() {
        BoxplotChartManager.initializeBoxplotChart('boxplot-chart-container');
    },
    
    showChartError: function() {
        const chartContainers = document.querySelectorAll('.boxplot-chart-container');
        chartContainers.forEach(container => {
            container.innerHTML = `
                <div class="p-8 text-center text-red-500">
                    <p>Chart rendering library not available. Charts cannot be displayed.</p>
                    <p class="text-sm mt-2">Please check your internet connection and refresh the page.</p>
                </div>`;
        });
    },
    
    fillBoxplotTable: function() {
        const tableBody = document.getElementById('boxplot-table-body');
        if (!tableBody) return;
        
        try {
            // Clear existing content
            tableBody.innerHTML = '';
            
            // Get models data
            const chartData = BoxplotChartManager.extractBoxplotData();
            if (!chartData || !chartData.models || chartData.models.length === 0) {
                this.showNoDataMessage(tableBody);
                return;
            }
            
            // Calculate stats for each model
            chartData.models.forEach(model => {
                const stats = BoxplotChartManager.calculateBoxplotStats(model.scores);
                if (!stats) return;
                
                const row = document.createElement('tr');
                
                // Model name
                const nameCell = document.createElement('td');
                nameCell.textContent = model.name;
                row.appendChild(nameCell);
                
                // Base score
                const baseScoreCell = document.createElement('td');
                baseScoreCell.textContent = model.baseScore.toFixed(4);
                row.appendChild(baseScoreCell);
                
                // Median
                const medianCell = document.createElement('td');
                medianCell.textContent = stats.median.toFixed(4);
                row.appendChild(medianCell);
                
                // Mean absolute deviation (approximate from quartiles)
                const madCell = document.createElement('td');
                const mad = stats.iqr / 1.35; // Approximate MAD from IQR
                madCell.textContent = mad.toFixed(4);
                row.appendChild(madCell);
                
                // Performance variance (IQR)
                const iqrCell = document.createElement('td');
                iqrCell.textContent = stats.iqr.toFixed(4);
                row.appendChild(iqrCell);
                
                // Min score
                const minCell = document.createElement('td');
                minCell.textContent = stats.min.toFixed(4);
                row.appendChild(minCell);
                
                // Max score
                const maxCell = document.createElement('td');
                maxCell.textContent = stats.max.toFixed(4);
                row.appendChild(maxCell);
                
                // Outliers
                const outliersCell = document.createElement('td');
                outliersCell.textContent = stats.outliers.length;
                row.appendChild(outliersCell);
                
                // Score drop (base to median)
                const dropCell = document.createElement('td');
                const drop = ((model.baseScore - stats.median) / model.baseScore * 100);
                dropCell.textContent = drop.toFixed(2) + '%';
                // Add a color indicator for drop
                if (drop > 5) {
                    dropCell.classList.add('text-red-600', 'font-semibold');
                } else if (drop > 0) {
                    dropCell.classList.add('text-yellow-600');
                } else {
                    dropCell.classList.add('text-green-600');
                }
                row.appendChild(dropCell);
                
                tableBody.appendChild(row);
            });
            
            // Add a hint about the table data
            const tableContainer = tableBody.closest('.table-container');
            if (tableContainer) {
                const hint = document.createElement('div');
                hint.className = 'text-sm text-gray-600 mt-2';
                hint.innerHTML = `
                    <p>This table shows key statistics for each model's performance under perturbation tests. 
                    A smaller performance drop and IQR indicates better robustness.</p>
                `;
                tableContainer.appendChild(hint);
            }
            
        } catch (error) {
            console.error("Error filling boxplot table:", error);
            this.showTableError(tableBody);
        }
    },
    
    updateBoxplotTable: function(selectedModel) {
        // Implementation for filtering the table by selected model
        const tableBody = document.getElementById('boxplot-table-body');
        if (!tableBody) return;
        
        // If 'all' is selected, show all models
        if (selectedModel === 'all') {
            this.fillBoxplotTable();
            return;
        }
        
        try {
            // Clear existing content
            tableBody.innerHTML = '';
            
            // Get models data
            const chartData = BoxplotChartManager.extractBoxplotData();
            if (!chartData || !chartData.models || chartData.models.length === 0) {
                this.showNoDataMessage(tableBody);
                return;
            }
            
            // Find selected model
            const model = chartData.models.find(m => m.name === selectedModel);
            if (!model) {
                this.showNoDataMessage(tableBody);
                return;
            }
            
            // Calculate stats for selected model
            const stats = BoxplotChartManager.calculateBoxplotStats(model.scores);
            if (!stats) {
                this.showNoDataMessage(tableBody);
                return;
            }
            
            const row = document.createElement('tr');
            
            // Model name
            const nameCell = document.createElement('td');
            nameCell.textContent = model.name;
            row.appendChild(nameCell);
            
            // Base score
            const baseScoreCell = document.createElement('td');
            baseScoreCell.textContent = model.baseScore.toFixed(4);
            row.appendChild(baseScoreCell);
            
            // Median
            const medianCell = document.createElement('td');
            medianCell.textContent = stats.median.toFixed(4);
            row.appendChild(medianCell);
            
            // Mean absolute deviation (approximate from quartiles)
            const madCell = document.createElement('td');
            const mad = stats.iqr / 1.35; // Approximate MAD from IQR
            madCell.textContent = mad.toFixed(4);
            row.appendChild(madCell);
            
            // Performance variance (IQR)
            const iqrCell = document.createElement('td');
            iqrCell.textContent = stats.iqr.toFixed(4);
            row.appendChild(iqrCell);
            
            // Min score
            const minCell = document.createElement('td');
            minCell.textContent = stats.min.toFixed(4);
            row.appendChild(minCell);
            
            // Max score
            const maxCell = document.createElement('td');
            maxCell.textContent = stats.max.toFixed(4);
            row.appendChild(maxCell);
            
            // Outliers
            const outliersCell = document.createElement('td');
            outliersCell.textContent = stats.outliers.length;
            row.appendChild(outliersCell);
            
            // Score drop (base to median)
            const dropCell = document.createElement('td');
            const drop = ((model.baseScore - stats.median) / model.baseScore * 100);
            dropCell.textContent = drop.toFixed(2) + '%';
            // Add a color indicator for drop
            if (drop > 5) {
                dropCell.classList.add('text-red-600', 'font-semibold');
            } else if (drop > 0) {
                dropCell.classList.add('text-yellow-600');
            } else {
                dropCell.classList.add('text-green-600');
            }
            row.appendChild(dropCell);
            
            tableBody.appendChild(row);
            
            // Add detailed statistics for the selected model
            this.addDetailedModelStats(tableBody, model, stats);
            
        } catch (error) {
            console.error("Error updating boxplot table:", error);
            this.showTableError(tableBody);
        }
    },
    
    /**
     * Add detailed statistics for a single model
     * @param {HTMLElement} tableBody - Table body element
     * @param {Object} model - Model data
     * @param {Object} stats - Boxplot statistics
     */
    addDetailedModelStats: function(tableBody, model, stats) {
        // Add a detailed stats row
        const detailRow = document.createElement('tr');
        detailRow.className = 'bg-gray-50';
        
        const detailCell = document.createElement('td');
        detailCell.colSpan = 9;
        detailCell.className = 'p-4';
        
        // Calculate additional statistics
        const meanScore = model.scores.reduce((sum, score) => sum + score, 0) / model.scores.length;
        const variance = model.scores.reduce((sum, score) => sum + Math.pow(score - meanScore, 2), 0) / model.scores.length;
        const stdDev = Math.sqrt(variance);
        const robustScore = 1 - (stats.iqr / model.baseScore);
        
        // Generate histogram data for score distribution
        const histogramData = this.generateHistogramData(model.scores, 10);
        
        // Create detailed stats content
        detailCell.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <h4 class="font-semibold mb-2">Detailed Statistics for ${model.name}</h4>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div>Model Type:</div>
                        <div class="font-medium">${model.modelType}</div>
                        
                        <div>Mean Score:</div>
                        <div class="font-medium">${meanScore.toFixed(4)}</div>
                        
                        <div>Standard Deviation:</div>
                        <div class="font-medium">${stdDev.toFixed(4)}</div>
                        
                        <div>Coefficient of Variation:</div>
                        <div class="font-medium">${(stdDev/meanScore).toFixed(4)}</div>
                        
                        <div>Estimated Robustness Score:</div>
                        <div class="font-medium">${robustScore.toFixed(4)}</div>
                        
                        <div>Total Samples:</div>
                        <div class="font-medium">${model.scores.length}</div>
                    </div>
                </div>
                <div>
                    <h4 class="font-semibold mb-2">Score Distribution</h4>
                    <div class="distribution-bar flex h-24 items-end">
                        ${this.renderDistributionBars(histogramData)}
                    </div>
                    <div class="text-xs mt-1 grid grid-cols-${histogramData.length} text-center">
                        ${histogramData.map(bin => `<div>${bin.range[0].toFixed(2)}</div>`).join('')}
                        <div>${histogramData[histogramData.length-1].range[1].toFixed(2)}</div>
                    </div>
                    <div class="flex justify-between text-xs mt-2">
                        <span>Min: ${stats.min.toFixed(4)}</span>
                        <span>Base: ${model.baseScore.toFixed(4)}</span>
                        <span>Max: ${stats.max.toFixed(4)}</span>
                    </div>
                </div>
            </div>
        `;
        
        detailRow.appendChild(detailCell);
        tableBody.appendChild(detailRow);
    },
    
    /**
     * Generate histogram data from scores
     * @param {Array} scores - Array of score values
     * @param {number} binCount - Number of bins
     * @returns {Array} Histogram data
     */
    generateHistogramData: function(scores, binCount) {
        if (!scores || scores.length === 0) return [];
        
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const binWidth = (max - min) / binCount;
        
        // Create bins
        const bins = Array(binCount).fill(0).map((_, i) => ({
            range: [min + (i * binWidth), min + ((i + 1) * binWidth)],
            count: 0
        }));
        
        // Fill bins
        scores.forEach(score => {
            // Handle edge case for max value
            if (score === max) {
                bins[binCount - 1].count++;
                return;
            }
            
            const binIndex = Math.floor((score - min) / binWidth);
            bins[binIndex].count++;
        });
        
        // Calculate max count for normalization
        const maxCount = Math.max(...bins.map(bin => bin.count));
        
        // Normalize counts
        bins.forEach(bin => {
            bin.height = maxCount > 0 ? (bin.count / maxCount) * 100 : 0;
        });
        
        return bins;
    },
    
    /**
     * Render distribution bar chart
     * @param {Array} histogramData - Histogram data
     * @returns {string} HTML for distribution bars
     */
    renderDistributionBars: function(histogramData) {
        return histogramData.map(bin => {
            return `
                <div class="px-1 flex-1">
                    <div class="bg-blue-500 opacity-70 hover:opacity-100 transition-opacity" 
                         style="height: ${bin.height}%; width: 100%;" 
                         title="${bin.count} scores between ${bin.range[0].toFixed(3)} and ${bin.range[1].toFixed(3)}">
                    </div>
                </div>
            `;
        }).join('');
    },
    
    showNoDataMessage: function(tableBody) {
        tableBody.innerHTML = '';
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 9;
        cell.innerHTML = `
            <div class="py-8 text-center">
                <p class="font-semibold mb-2">No boxplot data available</p>
                <p class="text-sm text-gray-600">Run robustness test with iterations to see boxplot distribution data.</p>
            </div>
        `;
        cell.style.textAlign = 'center';
        row.appendChild(cell);
        tableBody.appendChild(row);
    },
    
    showTableError: function(tableBody) {
        tableBody.innerHTML = '';
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 9;
        cell.innerHTML = `
            <div class="py-4 text-center text-red-500">
                <p>Error loading boxplot data</p>
                <p class="text-sm mt-1">Please try refreshing the page.</p>
            </div>
        `;
        row.appendChild(cell);
        tableBody.appendChild(row);
    }
};