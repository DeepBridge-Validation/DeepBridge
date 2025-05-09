/**
 * Direct Perturbation Handler
 * 
 * A simplified implementation to directly extract and display real perturbation data
 * without any synthetic values or complex dependencies
 */
(function() {
    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Check if the perturbation results container exists
        const container = document.getElementById('perturbation-results-container');
        if (container) {
            initializePerturbationResults();
        }
    });

    /**
     * Initialize perturbation results display
     */
    function initializePerturbationResults() {
        console.log("Initializing direct perturbation results display");
        
        try {
            // Extract data from report
            const data = extractPerturbationData();
            if (!data || !data.results || data.results.length === 0) {
                showError("No perturbation data available in the report");
                return;
            }
            
            // Render the perturbation analysis UI
            renderPerturbationAnalysis(data);
        } catch (error) {
            console.error("Error initializing perturbation results:", error);
            showError("Error processing perturbation data: " + error.message);
        }
    }
    
    /**
     * Extract perturbation data from the report data
     * Only extracts real data, no synthetic values
     */
    function extractPerturbationData() {
        if (!window.reportData || !window.reportData.raw || !window.reportData.raw.by_level) {
            console.error("No raw perturbation data available");
            return null;
        }
        
        const raw = window.reportData.raw;
        const byLevel = raw.by_level;
        
        // Basic result structure
        const result = {
            modelName: window.reportData.model_name || 'Model',
            modelType: window.reportData.model_type || 'Unknown',
            metric: window.reportData.metric || 'Score',
            baseScore: window.reportData.base_score || 0,
            results: []
        };
        
        // Get perturbation levels and sort them
        const levels = Object.keys(byLevel)
            .map(level => parseFloat(level))
            .filter(level => !isNaN(level))
            .sort((a, b) => a - b);
        
        console.log(`Found ${levels.length} perturbation levels: ${levels.join(', ')}`);
        
        // Process each level
        levels.forEach(level => {
            const levelData = byLevel[level.toString()];
            
            // Skip if no overall results
            if (!levelData || !levelData.overall_result) {
                console.log(`No overall result for level ${level}`);
                return;
            }
            
            const levelResult = {
                level: level,
                allFeatures: extractFeatureData(levelData, 'all_features', result.baseScore),
                featureSubset: extractFeatureData(levelData, 'feature_subset', result.baseScore)
            };
            
            // Add iteration scores from runs data if available
            if (levelData.runs) {
                if (levelData.runs.all_features && levelData.runs.all_features.length > 0) {
                    levelResult.allFeatures.iterations = extractIterationScores(levelData.runs.all_features);
                }
                
                if (levelData.runs.feature_subset && levelData.runs.feature_subset.length > 0) {
                    levelResult.featureSubset.iterations = extractIterationScores(levelData.runs.feature_subset);
                }
            }
            
            result.results.push(levelResult);
        });
        
        return result;
    }
    
    /**
     * Extract feature data from level results
     */
    function extractFeatureData(levelData, featureType, baseScore) {
        if (!levelData.overall_result || !levelData.overall_result[featureType]) {
            return null;
        }
        
        const featureResult = levelData.overall_result[featureType];
        
        return {
            baseScore: baseScore,
            meanScore: featureResult.mean_score || 0,
            worstScore: featureResult.worst_score || 0,
            impact: baseScore > 0 ? (baseScore - featureResult.mean_score) / baseScore : 0,
            iterations: [] // Will be populated if iteration data is available
        };
    }
    
    /**
     * Extract iteration scores from runs data
     */
    function extractIterationScores(runsData) {
        const iterationScores = [];
        
        runsData.forEach(run => {
            if (run.iterations && run.iterations.scores && Array.isArray(run.iterations.scores)) {
                // Add all scores from this run to the result
                iterationScores.push(...run.iterations.scores);
            }
        });
        
        return iterationScores;
    }
    
    /**
     * Render the perturbation analysis UI with the extracted data
     */
    function renderPerturbationAnalysis(data) {
        const container = document.getElementById('perturbation-results-container');
        if (!container) return;
        
        // Clear the container
        container.innerHTML = '';
        
        // Create main container
        const mainDiv = document.createElement('div');
        mainDiv.className = 'bg-white rounded-lg shadow-md overflow-hidden';
        
        // Add header
        mainDiv.appendChild(createHeader(data));
        
        // Create tabs container
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'p-4';
        
        // Add tabs navigation
        const tabsNav = document.createElement('div');
        tabsNav.className = 'tabs-nav flex border-b mb-4';
        
        const summaryTab = document.createElement('button');
        summaryTab.className = 'tab-button active px-4 py-2 mr-2 font-medium';
        summaryTab.textContent = 'Summary';
        summaryTab.dataset.tab = 'summary';
        
        const iterationsTab = document.createElement('button');
        iterationsTab.className = 'tab-button px-4 py-2 mr-2 font-medium';
        iterationsTab.textContent = 'Iterations';
        iterationsTab.dataset.tab = 'iterations';
        
        tabsNav.appendChild(summaryTab);
        tabsNav.appendChild(iterationsTab);
        tabsContainer.appendChild(tabsNav);
        
        // Add tab content containers
        const summaryContent = document.createElement('div');
        summaryContent.className = 'tab-content active';
        summaryContent.id = 'summary-tab';
        
        const iterationsContent = document.createElement('div');
        iterationsContent.className = 'tab-content hidden';
        iterationsContent.id = 'iterations-tab';
        
        // Populate summary tab with level selection and feature cards
        const summaryDiv = createSummaryContent(data);
        summaryContent.appendChild(summaryDiv);
        
        // Populate iterations tab with detailed data
        const iterationsDiv = createIterationsContent(data);
        iterationsContent.appendChild(iterationsDiv);
        
        // Add tab content to tabs container
        tabsContainer.appendChild(summaryContent);
        tabsContainer.appendChild(iterationsContent);
        
        // Add tabs container to main div
        mainDiv.appendChild(tabsContainer);
        
        // Add footer
        mainDiv.appendChild(createFooter(data));
        
        // Add the main div to the container
        container.appendChild(mainDiv);
        
        // Add event listeners for tab switching
        setupTabEventListeners();
        
        console.log("Perturbation analysis UI rendered with real data");
    }
    
    /**
     * Create header section with level selector
     */
    function createHeader(data) {
        const header = document.createElement('div');
        header.className = 'p-4 border-b border-gray-200';
        
        // Title
        const title = document.createElement('h3');
        title.className = 'text-xl font-medium mb-4';
        title.textContent = 'Perturbation Test Results';
        header.appendChild(title);
        
        // Level selector
        const levelSelector = document.createElement('div');
        levelSelector.className = 'mt-2';
        
        const levelLabel = document.createElement('label');
        levelLabel.className = 'block text-sm font-medium mb-2';
        levelLabel.textContent = 'Perturbation Level:';
        levelSelector.appendChild(levelLabel);
        
        const levelButtonGroup = document.createElement('div');
        levelButtonGroup.className = 'flex flex-wrap gap-2';
        
        // Add buttons for each perturbation level
        data.results.forEach((result, index) => {
            const button = document.createElement('button');
            button.className = 'level-btn px-3 py-1 rounded text-sm ' + 
                (index === 0 ? 'bg-blue-600 text-white' : 'bg-gray-200 hover:bg-gray-300 text-gray-700');
            button.textContent = `${result.level * 100}%`;
            button.dataset.level = result.level;
            levelButtonGroup.appendChild(button);
        });
        
        levelSelector.appendChild(levelButtonGroup);
        header.appendChild(levelSelector);
        
        return header;
    }
    
    /**
     * Create summary tab content
     */
    function createSummaryContent(data) {
        const summaryDiv = document.createElement('div');
        
        // Default to first level results
        const selectedLevel = data.results[0].level; 
        const selectedData = data.results[0];
        
        // Feature summaries grid
        const summaryGrid = document.createElement('div');
        summaryGrid.className = 'grid grid-cols-1 md:grid-cols-2 gap-4';
        
        // All Features Summary
        if (selectedData.allFeatures) {
            summaryGrid.appendChild(createFeatureSummary(
                selectedData.allFeatures, 
                'allFeatures', 
                'All Features'
            ));
        }
        
        // Feature Subset Summary
        if (selectedData.featureSubset) {
            summaryGrid.appendChild(createFeatureSummary(
                selectedData.featureSubset, 
                'featureSubset', 
                'Feature Subset'
            ));
        }
        
        summaryDiv.appendChild(summaryGrid);
        
        // Analysis box
        if (selectedData.allFeatures) {
            const analysisBox = document.createElement('div');
            analysisBox.className = 'mt-4 p-4 bg-blue-50 rounded-lg';
            
            const analysisTitle = document.createElement('h3');
            analysisTitle.className = 'font-medium text-blue-800 mb-2';
            analysisTitle.textContent = 'Analysis';
            analysisBox.appendChild(analysisTitle);
            
            const analysisParagraph = document.createElement('p');
            analysisParagraph.className = 'text-sm text-blue-700';
            
            // Generate analysis text
            let analysisText = `At ${selectedLevel * 100}% perturbation, the model shows `;
            if (selectedData.allFeatures.impact < 0) {
                analysisText += 'improvement ';
            } else {
                analysisText += `degradation of ${formatNumber(selectedData.allFeatures.impact * 100, 2)}% `;
            }
            analysisText += 'when all features are perturbed. ';
            
            if (selectedData.featureSubset && selectedData.featureSubset.impact < selectedData.allFeatures.impact) {
                analysisText += `The feature subset shows better robustness with only ${formatNumber(selectedData.featureSubset.impact * 100, 2)}% impact.`;
            }
            
            analysisParagraph.textContent = analysisText;
            analysisBox.appendChild(analysisParagraph);
            
            summaryDiv.appendChild(analysisBox);
        }
        
        return summaryDiv;
    }
    
    /**
     * Create feature summary panel
     */
    function createFeatureSummary(featureData, sectionId, title) {
        const summaryPanel = document.createElement('div');
        summaryPanel.className = 'border rounded-lg overflow-hidden shadow-sm';
        
        // Header
        const panelHeader = document.createElement('div');
        panelHeader.className = 'bg-gray-50 p-3 flex justify-between items-center';
        
        const panelTitle = document.createElement('h4');
        panelTitle.className = 'font-medium';
        panelTitle.textContent = title;
        panelHeader.appendChild(panelTitle);
        
        // Impact indicator
        const impactSpan = document.createElement('span');
        impactSpan.className = `text-sm font-semibold ${getImpactColorClass(featureData.impact)}`;
        impactSpan.textContent = `Impact: ${formatNumber(featureData.impact * 100)}%`;
        panelHeader.appendChild(impactSpan);
        
        summaryPanel.appendChild(panelHeader);
        
        // Content
        const panelContent = document.createElement('div');
        panelContent.className = 'p-4';
        
        // Stats table
        const statsTable = document.createElement('table');
        statsTable.className = 'min-w-full divide-y divide-gray-200';
        
        const tbody = document.createElement('tbody');
        tbody.className = 'divide-y divide-gray-200';
        
        // Add rows to the table
        const rows = [
            { label: 'Base Score', value: formatNumber(featureData.baseScore) },
            { 
                label: 'Mean Score',
                value: formatNumber(featureData.meanScore),
                diff: featureData.meanScore - featureData.baseScore
            },
            { 
                label: 'Worst Score',
                value: formatNumber(featureData.worstScore),
                diff: featureData.worstScore - featureData.baseScore
            }
        ];
        
        // Create rows
        rows.forEach(rowData => {
            const row = document.createElement('tr');
            
            const labelCell = document.createElement('td');
            labelCell.className = 'px-3 py-2 text-sm font-medium text-gray-900';
            labelCell.textContent = rowData.label;
            row.appendChild(labelCell);
            
            const valueCell = document.createElement('td');
            valueCell.className = 'px-3 py-2 text-sm text-gray-700';
            
            if (rowData.diff !== undefined) {
                const valueText = document.createTextNode(rowData.value);
                valueCell.appendChild(valueText);
                
                const diffSpan = document.createElement('span');
                diffSpan.className = `ml-2 ${rowData.diff >= 0 ? 'text-green-600' : 'text-red-600'}`;
                diffSpan.textContent = `(${rowData.diff >= 0 ? '+' : ''}${formatNumber(rowData.diff * 100, 2)}%)`;
                valueCell.appendChild(diffSpan);
            } else {
                valueCell.textContent = rowData.value;
            }
            
            row.appendChild(valueCell);
            tbody.appendChild(row);
        });
        
        statsTable.appendChild(tbody);
        panelContent.appendChild(statsTable);
        
        // Impact bar
        const barContainer = document.createElement('div');
        barContainer.className = 'mt-3';
        
        const barLabels = document.createElement('div');
        barLabels.className = 'flex justify-between text-xs text-gray-500 mb-1';
        
        const startLabel = document.createElement('span');
        startLabel.textContent = '0% (Base)';
        barLabels.appendChild(startLabel);
        
        const middleLabel = document.createElement('span');
        middleLabel.textContent = 'Impact';
        barLabels.appendChild(middleLabel);
        
        const endLabel = document.createElement('span');
        endLabel.textContent = '25%';
        barLabels.appendChild(endLabel);
        
        barContainer.appendChild(barLabels);
        
        const barBg = document.createElement('div');
        barBg.className = 'w-full bg-gray-200 rounded-full h-2.5';
        
        const barFill = document.createElement('div');
        barFill.className = `h-2.5 rounded-full ${featureData.impact < 0 ? 'bg-green-500' : 'bg-red-500'}`;
        barFill.style.width = `${Math.min(Math.abs(featureData.impact) * 100 * 4, 100)}%`;
        
        barBg.appendChild(barFill);
        barContainer.appendChild(barBg);
        
        panelContent.appendChild(barContainer);
        summaryPanel.appendChild(panelContent);
        
        return summaryPanel;
    }
    
    /**
     * Create iterations tab content with table of iteration data
     */
    function createIterationsContent(data) {
        const iterationsDiv = document.createElement('div');
        
        // Default to first level results
        const selectedData = data.results[0];
        
        // Simple message if no iterations
        if (!selectedData.allFeatures || !selectedData.allFeatures.iterations || 
            selectedData.allFeatures.iterations.length === 0) {
            const noDataMessage = document.createElement('p');
            noDataMessage.className = 'text-center text-gray-500 my-8';
            noDataMessage.textContent = 'No iteration data available for this perturbation level.';
            iterationsDiv.appendChild(noDataMessage);
            return iterationsDiv;
        }
        
        // Table container
        const tableContainer = document.createElement('div');
        tableContainer.className = 'overflow-x-auto';
        
        const table = document.createElement('table');
        table.className = 'min-w-full divide-y divide-gray-200';
        
        // Table header
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50';
        
        const headerRow = document.createElement('tr');
        
        const headers = [
            { text: 'Iteration', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider' },
            { text: 'All Features', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider' },
            { text: 'Feature Subset', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider' },
            { text: 'Difference', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider' }
        ];
        
        headers.forEach(header => {
            const th = document.createElement('th');
            th.className = header.className;
            th.textContent = header.text;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Table body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white divide-y divide-gray-200';
        
        // Base score row
        const baseRow = document.createElement('tr');
        baseRow.className = 'bg-gray-100';
        
        const baseIterationCell = document.createElement('td');
        baseIterationCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-bold text-gray-900';
        baseIterationCell.textContent = 'Base';
        baseRow.appendChild(baseIterationCell);
        
        const baseAllFeaturesCell = document.createElement('td');
        baseAllFeaturesCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        baseAllFeaturesCell.textContent = formatNumber(selectedData.allFeatures.baseScore);
        baseRow.appendChild(baseAllFeaturesCell);
        
        const baseFeatureSubsetCell = document.createElement('td');
        baseFeatureSubsetCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        baseFeatureSubsetCell.textContent = selectedData.featureSubset ? 
            formatNumber(selectedData.featureSubset.baseScore) : 'N/A';
        baseRow.appendChild(baseFeatureSubsetCell);
        
        const baseDifferenceCell = document.createElement('td');
        baseDifferenceCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        baseDifferenceCell.textContent = '0.0000';
        baseRow.appendChild(baseDifferenceCell);
        
        tbody.appendChild(baseRow);
        
        // Get iteration data
        const allFeaturesIterations = selectedData.allFeatures.iterations || [];
        const featureSubsetIterations = selectedData.featureSubset ? 
            (selectedData.featureSubset.iterations || []) : [];
        
        const maxIterations = Math.max(allFeaturesIterations.length, featureSubsetIterations.length);
        
        // Create a row for each iteration
        for (let i = 0; i < maxIterations; i++) {
            const iterationRow = document.createElement('tr');
            
            // Iteration number
            const iterationCell = document.createElement('td');
            iterationCell.className = 'px-6 py-3 whitespace-nowrap text-sm text-gray-900';
            iterationCell.textContent = `#${i + 1}`;
            iterationRow.appendChild(iterationCell);
            
            // All features score
            const allFeaturesScore = allFeaturesIterations[i];
            const allFeaturesCell = document.createElement('td');
            allFeaturesCell.className = 'px-6 py-3 whitespace-nowrap text-sm text-gray-700';
            
            if (allFeaturesScore !== undefined) {
                const scoreText = document.createTextNode(formatNumber(allFeaturesScore));
                allFeaturesCell.appendChild(scoreText);
                
                // Add difference if base score provided
                const diff = allFeaturesScore - selectedData.allFeatures.baseScore;
                const diffSpan = document.createElement('span');
                diffSpan.className = `ml-2 text-xs ${diff >= 0 ? 'text-green-600' : 'text-red-600'}`;
                diffSpan.textContent = `(${diff >= 0 ? '+' : ''}${formatNumber(diff * 100, 2)}%)`;
                allFeaturesCell.appendChild(diffSpan);
            } else {
                allFeaturesCell.textContent = 'N/A';
            }
            
            iterationRow.appendChild(allFeaturesCell);
            
            // Feature subset score
            const featureSubsetScore = featureSubsetIterations[i];
            const featureSubsetCell = document.createElement('td');
            featureSubsetCell.className = 'px-6 py-3 whitespace-nowrap text-sm text-gray-700';
            
            if (featureSubsetScore !== undefined && selectedData.featureSubset) {
                const scoreText = document.createTextNode(formatNumber(featureSubsetScore));
                featureSubsetCell.appendChild(scoreText);
                
                // Add difference if base score provided
                const diff = featureSubsetScore - selectedData.featureSubset.baseScore;
                const diffSpan = document.createElement('span');
                diffSpan.className = `ml-2 text-xs ${diff >= 0 ? 'text-green-600' : 'text-red-600'}`;
                diffSpan.textContent = `(${diff >= 0 ? '+' : ''}${formatNumber(diff * 100, 2)}%)`;
                featureSubsetCell.appendChild(diffSpan);
            } else {
                featureSubsetCell.textContent = 'N/A';
            }
            
            iterationRow.appendChild(featureSubsetCell);
            
            // Difference between scores
            const differenceCell = document.createElement('td');
            differenceCell.className = 'px-6 py-3 whitespace-nowrap text-sm';
            
            if (allFeaturesScore !== undefined && featureSubsetScore !== undefined) {
                const scoreDiff = featureSubsetScore - allFeaturesScore;
                differenceCell.className += ` ${scoreDiff > 0 ? 'text-green-600' : 'text-red-600'}`;
                differenceCell.textContent = `${scoreDiff > 0 ? '+' : ''}${formatNumber(scoreDiff)}`;
            } else {
                differenceCell.className += ' text-gray-500';
                differenceCell.textContent = 'N/A';
            }
            
            iterationRow.appendChild(differenceCell);
            tbody.appendChild(iterationRow);
        }
        
        // Mean row
        if (selectedData.allFeatures.meanScore !== undefined) {
            const meanRow = document.createElement('tr');
            meanRow.className = 'bg-gray-100';
            
            const meanLabelCell = document.createElement('td');
            meanLabelCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-bold text-gray-900';
            meanLabelCell.textContent = 'Mean';
            meanRow.appendChild(meanLabelCell);
            
            const allFeaturesMeanCell = document.createElement('td');
            allFeaturesMeanCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
            allFeaturesMeanCell.textContent = formatNumber(selectedData.allFeatures.meanScore);
            meanRow.appendChild(allFeaturesMeanCell);
            
            const featureSubsetMeanCell = document.createElement('td');
            featureSubsetMeanCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
            featureSubsetMeanCell.textContent = selectedData.featureSubset ? 
                formatNumber(selectedData.featureSubset.meanScore) : 'N/A';
            meanRow.appendChild(featureSubsetMeanCell);
            
            const meanDiffCell = document.createElement('td');
            
            if (selectedData.featureSubset && selectedData.featureSubset.meanScore !== undefined) {
                const meanDiff = selectedData.featureSubset.meanScore - selectedData.allFeatures.meanScore;
                meanDiffCell.className = `px-6 py-3 whitespace-nowrap text-sm font-medium ${
                    meanDiff > 0 ? 'text-green-600' : 'text-red-600'
                }`;
                meanDiffCell.textContent = `${meanDiff > 0 ? '+' : ''}${formatNumber(meanDiff)}`;
            } else {
                meanDiffCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-500';
                meanDiffCell.textContent = 'N/A';
            }
            
            meanRow.appendChild(meanDiffCell);
            tbody.appendChild(meanRow);
        }
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        iterationsDiv.appendChild(tableContainer);
        
        return iterationsDiv;
    }
    
    /**
     * Create footer section
     */
    function createFooter(data) {
        const footer = document.createElement('div');
        footer.className = 'p-3 border-t border-gray-200 text-xs text-gray-500';
        
        const footerFlex = document.createElement('div');
        footerFlex.className = 'flex justify-between';
        
        // Left section
        const leftSpan = document.createElement('span');
        leftSpan.textContent = `Perturbation Test • ${data.modelType || 'Model'} • ${data.metric || 'Score'} Metric`;
        footerFlex.appendChild(leftSpan);
        
        // Right section
        const rightSpan = document.createElement('span');
        rightSpan.textContent = `Base Score: ${formatNumber(data.baseScore)} • Date: ${new Date().toLocaleDateString()}`;
        footerFlex.appendChild(rightSpan);
        
        footer.appendChild(footerFlex);
        
        return footer;
    }
    
    /**
     * Setup event listeners for tab switching and level selection
     */
    function setupTabEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                    btn.classList.add('text-gray-500');
                    btn.classList.remove('text-blue-600', 'border-b-2', 'border-blue-500');
                });
                
                // Add active class to clicked button
                this.classList.add('active', 'text-blue-600', 'border-b-2', 'border-blue-500');
                this.classList.remove('text-gray-500');
                
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.add('hidden');
                    content.classList.remove('active');
                });
                
                // Show selected tab content
                const tabId = this.dataset.tab;
                const tabContent = document.getElementById(tabId + '-tab');
                if (tabContent) {
                    tabContent.classList.remove('hidden');
                    tabContent.classList.add('active');
                }
            });
        });
        
        // Level selection
        document.querySelectorAll('.level-btn').forEach(button => {
            button.addEventListener('click', function() {
                // Update active state
                document.querySelectorAll('.level-btn').forEach(btn => {
                    btn.classList.remove('bg-blue-600', 'text-white');
                    btn.classList.add('bg-gray-200', 'text-gray-700');
                });
                
                this.classList.remove('bg-gray-200', 'text-gray-700');
                this.classList.add('bg-blue-600', 'text-white');
                
                // Get selected level
                const selectedLevel = parseFloat(this.dataset.level);
                
                // Re-render the perturbation analysis with the selected level
                updatePerturbationAnalysisForLevel(selectedLevel);
            });
        });
    }
    
    /**
     * Update the perturbation analysis UI for a specific level
     */
    function updatePerturbationAnalysisForLevel(selectedLevel) {
        // Re-extract data
        const data = extractPerturbationData();
        if (!data || !data.results || data.results.length === 0) {
            showError("No perturbation data available");
            return;
        }
        
        // Find the data for the selected level
        const selectedData = data.results.find(result => result.level === selectedLevel);
        if (!selectedData) {
            console.error(`No data found for level ${selectedLevel}`);
            return;
        }
        
        // Update summary tab
        const summaryTab = document.getElementById('summary-tab');
        if (summaryTab) {
            summaryTab.innerHTML = '';
            summaryTab.appendChild(createSummaryContent({
                ...data,
                results: [selectedData] // Just include the selected level
            }));
        }
        
        // Update iterations tab
        const iterationsTab = document.getElementById('iterations-tab');
        if (iterationsTab) {
            iterationsTab.innerHTML = '';
            iterationsTab.appendChild(createIterationsContent({
                ...data,
                results: [selectedData] // Just include the selected level
            }));
        }
    }
    
    /**
     * Show error message in the container
     */
    function showError(message) {
        const container = document.getElementById('perturbation-results-container');
        if (!container) return;
        
        container.innerHTML = `
            <div class="p-6 text-center bg-red-50 rounded-lg border border-red-200">
                <div class="text-4xl mb-3 text-red-500">❌</div>
                <h3 class="text-lg font-medium text-red-800 mb-2">Error Loading Perturbation Results</h3>
                <p class="text-red-600">${message || 'An unknown error occurred'}</p>
            </div>
        `;
    }
    
    /**
     * Format a number for display
     */
    function formatNumber(value, decimals = 4) {
        if (value === undefined || value === null || isNaN(value)) {
            return 'N/A';
        }
        return value.toFixed(decimals);
    }
    
    /**
     * Get color class for impact value
     */
    function getImpactColorClass(impact) {
        if (impact < 0) return 'text-green-600';
        if (impact < 0.05) return 'text-yellow-600';
        if (impact < 0.1) return 'text-orange-600';
        return 'text-red-600';
    }
})();