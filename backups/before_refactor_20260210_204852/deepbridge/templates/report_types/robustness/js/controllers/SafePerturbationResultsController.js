// Safe implementation of the PerturbationResultsController
// This version fixes syntax errors and adds error handling
(function() {
    // Create the controller if it doesn't exist
    window.SafePerturbationResultsController = {
        // Store the extracted data
        data: null,
        
        // Store UI state
        state: {
            selectedLevel: null,
            activeTab: 'summary',
            expandedSection: 'allFeatures'
        },
        
        /**
         * Initialize the controller safely
         */
        init: function() {
            console.log("Initializing Safe Perturbation Results Controller");
            
            try {
                // Extract perturbation data if the manager exists
                if (typeof PerturbationResultsManager !== 'undefined') {
                    this.data = PerturbationResultsManager.extractPerturbationData();
                    
                    // Set initial selected level to middle value if available, otherwise first value
                    if (this.data && this.data.results && this.data.results.length > 0) {
                        const middleIndex = Math.floor(this.data.results.length / 2);
                        this.state.selectedLevel = this.data.results[middleIndex].level;
                    }
                    
                    // Render the component
                    this.render();
                    
                    // Initialize event listeners
                    this.initEventListeners();
                } else {
                    console.error("PerturbationResultsManager not found");
                    this.renderError("Unable to extract perturbation data. PerturbationResultsManager not available.");
                }
            } catch (error) {
                console.error("Error initializing PerturbationResultsController:", error);
                this.renderError("An error occurred while initializing: " + error.message);
            }
        },
        
        /**
         * Format a number safely
         * @param {number} value - Number to format
         * @param {number} decimals - Decimal places
         * @returns {string} Formatted number
         */
        formatNumber: function(value, decimals) {
            try {
                if (typeof value !== 'number') return 'N/A';
                const dec = decimals || 4;
                return value.toFixed(dec);
            } catch (error) {
                return 'Error';
            }
        },
        
        /**
         * Get color class for impact value safely
         * @param {number} impact - Impact value
         * @returns {string} CSS class
         */
        getImpactColorClass: function(impact) {
            try {
                if (impact < 0) return 'text-green-600';
                if (impact < 0.05) return 'text-yellow-600';
                if (impact < 0.1) return 'text-orange-600';
                return 'text-red-600';
            } catch (error) {
                return 'text-gray-600';
            }
        },
        
        /**
         * Get background color class for score comparison
         * @param {number} score - Score to compare
         * @param {number} baseScore - Base score
         * @returns {string} CSS class
         */
        getScoreBgColorClass: function(score, baseScore) {
            try {
                if (!score || !baseScore) return '';
                const diff = score - baseScore;
                if (diff > 0) return 'bg-green-50';
                if (diff < -0.1) return 'bg-red-50';
                if (diff < -0.05) return 'bg-orange-50';
                if (diff < 0) return 'bg-yellow-50';
                return '';
            } catch (error) {
                return '';
            }
        },
        
        /**
         * Create error message element
         * @param {string} message - Error message to display
         * @returns {HTMLElement} Error message element
         */
        createErrorMessage: function(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'p-8 text-center text-red-500';
            errorDiv.textContent = message || 'An error occurred loading perturbation data';
            return errorDiv;
        },
        
        /**
         * Render error message
         * @param {string} message - Error message to display
         */
        renderError: function(message) {
            const container = document.getElementById('perturbation-results-container');
            if (!container) {
                console.error("Perturbation results container not found");
                return;
            }
            
            // Clear container
            container.innerHTML = '';
            
            // Create error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'p-6 text-center text-red-500 bg-red-50 rounded-lg border border-red-200';
            
            const errorIcon = document.createElement('div');
            errorIcon.className = 'text-4xl mb-3';
            errorIcon.innerHTML = '❌';
            errorDiv.appendChild(errorIcon);
            
            const errorTitle = document.createElement('h3');
            errorTitle.className = 'text-lg font-medium text-red-800 mb-2';
            errorTitle.textContent = 'Error Loading Perturbation Results';
            errorDiv.appendChild(errorTitle);
            
            const errorMessage = document.createElement('p');
            errorMessage.className = 'text-red-600';
            errorMessage.textContent = message || 'An unknown error occurred while loading perturbation data.';
            errorDiv.appendChild(errorMessage);
            
            container.appendChild(errorDiv);
        },
        
        /**
         * Initialize event listeners
         */
        initEventListeners: function() {
            document.addEventListener('click', (event) => {
                try {
                    // Handle level selection
                    if (event.target.classList.contains('level-btn')) {
                        const level = parseFloat(event.target.dataset.level);
                        this.state.selectedLevel = level;
                        this.render();
                    }
                    
                    // Handle tab selection
                    if (event.target.classList.contains('tab-btn')) {
                        const tab = event.target.dataset.tab;
                        this.state.activeTab = tab;
                        this.render();
                    }
                    
                    // Handle section toggle
                    if (event.target.closest('.section-toggle')) {
                        const section = event.target.closest('.section-toggle').dataset.section;
                        this.state.expandedSection = this.state.expandedSection === section ? null : section;
                        this.render();
                    }
                } catch (error) {
                    console.error("Error in event listener:", error);
                }
            });
        },
        
        /**
         * Render the perturbation results UI
         */
        render: function() {
            try {
                const container = document.getElementById('perturbation-results-container');
                if (!container) {
                    console.error("Perturbation results container not found");
                    return;
                }
                
                // Clear container
                container.innerHTML = '';
                
                // Create main container
                const mainDiv = document.createElement('div');
                mainDiv.className = 'bg-white rounded-lg shadow-md overflow-hidden';
                
                // Create header
                mainDiv.appendChild(this.createHeader());
                
                // Create content based on active tab
                if (this.state.activeTab === 'summary') {
                    mainDiv.appendChild(this.createSummaryTab());
                } else {
                    mainDiv.appendChild(this.createIterationsTab());
                }
                
                // Create footer
                mainDiv.appendChild(this.createFooter());
                
                // Add to container
                container.appendChild(mainDiv);
            } catch (error) {
                console.error("Error rendering perturbation results:", error);
                this.renderError(error.message);
            }
        },
        
        // Create header section (same implementation with added try-catch)
        createHeader: function() {
            try {
                const header = document.createElement('div');
                header.className = 'p-4 border-b border-gray-200';
                
                // Title
                const title = document.createElement('h2');
                title.className = 'text-xl font-bold text-gray-800';
                title.textContent = 'Perturbation Test Results';
                header.appendChild(title);
                
                // Perturbation level selector (only if we have data)
                if (this.data && this.data.results && this.data.results.length > 0) {
                    const levelSelectorContainer = document.createElement('div');
                    levelSelectorContainer.className = 'mt-3';
                    
                    const levelLabel = document.createElement('label');
                    levelLabel.className = 'block text-sm font-medium text-gray-700 mb-1';
                    levelLabel.textContent = 'Perturbation Level:';
                    levelSelectorContainer.appendChild(levelLabel);
                    
                    const buttonGroup = document.createElement('div');
                    buttonGroup.className = 'flex space-x-1';
                    
                    this.data.results.forEach(result => {
                        const levelBtn = document.createElement('button');
                        levelBtn.className = `level-btn px-3 py-1 rounded text-sm ${
                            this.state.selectedLevel === result.level
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                        }`;
                        levelBtn.textContent = `${result.level * 100}%`;
                        levelBtn.dataset.level = result.level;
                        buttonGroup.appendChild(levelBtn);
                    });
                    
                    levelSelectorContainer.appendChild(buttonGroup);
                    header.appendChild(levelSelectorContainer);
                }
                
                // Tabs
                const tabsContainer = document.createElement('div');
                tabsContainer.className = 'mt-3 border-b border-gray-200';
                
                const tabsNav = document.createElement('nav');
                tabsNav.className = '-mb-px flex space-x-8';
                
                const summaryTab = document.createElement('button');
                summaryTab.className = `tab-btn whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                    this.state.activeTab === 'summary'
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                }`;
                summaryTab.textContent = 'Summary';
                summaryTab.dataset.tab = 'summary';
                tabsNav.appendChild(summaryTab);
                
                const iterationsTab = document.createElement('button');
                iterationsTab.className = `tab-btn whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
                    this.state.activeTab === 'iterations'
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                }`;
                iterationsTab.textContent = 'Iterations';
                iterationsTab.dataset.tab = 'iterations';
                tabsNav.appendChild(iterationsTab);
                
                tabsContainer.appendChild(tabsNav);
                header.appendChild(tabsContainer);
                
                return header;
            } catch (error) {
                console.error("Error creating header:", error);
                return this.createErrorMessage("Error creating header");
            }
        },
        
        // Create summary tab content (wrapper with try-catch)
        createSummaryTab: function() {
            try {
                // Find selected data
                const selectedData = this.data && this.data.results ? 
                    this.data.results.find(r => r.level === this.state.selectedLevel) : null;
                
                if (!selectedData) {
                    return this.createErrorMessage("No data available for selected level");
                }
                
                const summaryTab = document.createElement('div');
                summaryTab.className = 'p-4';
                
                // Feature summaries grid
                const summaryGrid = document.createElement('div');
                summaryGrid.className = 'grid grid-cols-1 md:grid-cols-2 gap-4';
                
                // All Features Summary
                if (selectedData.allFeatures) {
                    summaryGrid.appendChild(this.createFeatureSummary(
                        selectedData.allFeatures, 
                        'allFeatures', 
                        'All Features'
                    ));
                }
                
                // Feature Subset Summary
                if (selectedData.featureSubset) {
                    summaryGrid.appendChild(this.createFeatureSummary(
                        selectedData.featureSubset, 
                        'featureSubset', 
                        'Feature Subset'
                    ));
                }
                
                summaryTab.appendChild(summaryGrid);
                
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
                    
                    // Generate analysis text safely
                    let analysisText = `At ${this.state.selectedLevel * 100}% perturbation, the model shows `;
                    if (selectedData.allFeatures.impact < 0) {
                        analysisText += 'improvement ';
                    } else {
                        analysisText += `degradation of ${this.formatNumber(selectedData.allFeatures.impact * 100, 2)}% `;
                    }
                    analysisText += 'when all features are perturbed. ';
                    
                    if (selectedData.featureSubset && selectedData.featureSubset.impact < selectedData.allFeatures.impact) {
                        analysisText += `The feature subset shows better robustness with only ${this.formatNumber(selectedData.featureSubset.impact * 100, 2)}% impact.`;
                    }
                    
                    analysisParagraph.textContent = analysisText;
                    analysisBox.appendChild(analysisParagraph);
                    
                    summaryTab.appendChild(analysisBox);
                }
                
                return summaryTab;
                
            } catch (error) {
                console.error("Error creating summary tab:", error);
                return this.createErrorMessage("Error creating summary tab: " + error.message);
            }
        },
        
        // Create feature summary panel (with added try-catch)
        createFeatureSummary: function(featureData, sectionId, title) {
            try {
                if (!featureData) {
                    return this.createErrorMessage(`No data available for ${title}`);
                }
                
                const summaryPanel = document.createElement('div');
                summaryPanel.className = `border rounded-lg overflow-hidden ${
                    this.state.expandedSection === sectionId ? 'shadow-md' : ''
                }`;
                
                // Header
                const panelHeader = document.createElement('div');
                panelHeader.className = 'section-toggle bg-gray-50 p-3 flex justify-between items-center cursor-pointer';
                panelHeader.dataset.section = sectionId;
                
                const panelTitle = document.createElement('h3');
                panelTitle.className = 'font-medium';
                panelTitle.textContent = title;
                panelHeader.appendChild(panelTitle);
                
                const impactSpan = document.createElement('span');
                impactSpan.className = `text-sm font-semibold ${this.getImpactColorClass(featureData.impact)}`;
                impactSpan.textContent = `Impact: ${this.formatNumber(featureData.impact * 100)}%`;
                panelHeader.appendChild(impactSpan);
                
                summaryPanel.appendChild(panelHeader);
                
                // Content (only if expanded)
                if (this.state.expandedSection === sectionId) {
                    const panelContent = document.createElement('div');
                    panelContent.className = 'p-4';
                    
                    // Stats table
                    const statsTable = document.createElement('table');
                    statsTable.className = 'min-w-full divide-y divide-gray-200';
                    
                    const tbody = document.createElement('tbody');
                    tbody.className = 'divide-y divide-gray-200';
                    
                    // Add rows to the table
                    const rows = [
                        { label: 'Base Score', value: this.formatNumber(featureData.baseScore) },
                        { 
                            label: 'Mean Score',
                            value: this.formatNumber(featureData.meanScore),
                            diff: featureData.meanScore - featureData.baseScore
                        },
                        { 
                            label: 'Worst Score',
                            value: this.formatNumber(featureData.worstScore),
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
                            diffSpan.textContent = `(${rowData.diff >= 0 ? '+' : ''}${this.formatNumber(rowData.diff * 100, 2)}%)`;
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
                }
                
                return summaryPanel;
                
            } catch (error) {
                console.error(`Error creating feature summary for ${title}:`, error);
                return this.createErrorMessage(`Error creating ${title} summary`);
            }
        },
        
        // Create iterations tab with try-catch
        createIterationsTab: function() {
            try {
                // Find selected data
                const selectedData = this.data && this.data.results ? 
                    this.data.results.find(r => r.level === this.state.selectedLevel) : null;
                
                if (!selectedData) {
                    return this.createErrorMessage("No data available for selected level");
                }
                
                const iterationsTab = document.createElement('div');
                iterationsTab.className = 'p-4';
                
                // Simple message if no iterations
                if (!selectedData.allFeatures || !selectedData.allFeatures.iterations || 
                    selectedData.allFeatures.iterations.length === 0) {
                    const noDataMessage = document.createElement('p');
                    noDataMessage.className = 'text-center text-gray-500 my-8';
                    noDataMessage.textContent = 'No iteration data available for this perturbation level.';
                    iterationsTab.appendChild(noDataMessage);
                    return iterationsTab;
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
                
                // Simple function to handle all output of scores
                const createScoreCell = (score, baseScore, className) => {
                    const cell = document.createElement('td');
                    cell.className = className || 'px-6 py-3 whitespace-nowrap text-sm text-gray-700';
                    
                    if (score === null || score === undefined) {
                        cell.textContent = 'N/A';
                        return cell;
                    }
                    
                    const scoreText = document.createTextNode(this.formatNumber(score));
                    cell.appendChild(scoreText);
                    
                    // Add difference if base score provided
                    if (baseScore !== undefined && baseScore !== null) {
                        const diff = score - baseScore;
                        const diffSpan = document.createElement('span');
                        diffSpan.className = `ml-2 text-xs ${diff >= 0 ? 'text-green-600' : 'text-red-600'}`;
                        diffSpan.textContent = `(${diff >= 0 ? '+' : ''}${this.formatNumber(diff * 100, 2)}%)`;
                        cell.appendChild(diffSpan);
                    }
                    
                    return cell;
                };
                
                // Base score row
                const baseRow = document.createElement('tr');
                baseRow.className = 'bg-gray-100';
                
                const baseIterationCell = document.createElement('td');
                baseIterationCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-bold text-gray-900';
                baseIterationCell.textContent = 'Base';
                baseRow.appendChild(baseIterationCell);
                
                baseRow.appendChild(createScoreCell(
                    selectedData.allFeatures.baseScore, 
                    null,
                    'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900'
                ));
                
                baseRow.appendChild(createScoreCell(
                    selectedData.featureSubset ? selectedData.featureSubset.baseScore : null,
                    null,
                    'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900'
                ));
                
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
                    const allFeaturesScore = allFeaturesIterations[i] || null;
                    const allFeaturesBaseScore = selectedData.allFeatures.baseScore;
                    iterationRow.appendChild(createScoreCell(
                        allFeaturesScore, 
                        allFeaturesBaseScore,
                        `px-6 py-3 whitespace-nowrap text-sm text-gray-700 ${
                            this.getScoreBgColorClass(allFeaturesScore, allFeaturesBaseScore)
                        }`
                    ));
                    
                    // Feature subset score
                    const featureSubsetScore = featureSubsetIterations[i] || null;
                    const featureSubsetBaseScore = selectedData.featureSubset ? 
                        selectedData.featureSubset.baseScore : null;
                    iterationRow.appendChild(createScoreCell(
                        featureSubsetScore,
                        featureSubsetBaseScore,
                        `px-6 py-3 whitespace-nowrap text-sm text-gray-700 ${
                            this.getScoreBgColorClass(featureSubsetScore, featureSubsetBaseScore)
                        }`
                    ));
                    
                    // Difference between scores
                    if (allFeaturesScore !== null && featureSubsetScore !== null) {
                        const scoreDiff = featureSubsetScore - allFeaturesScore;
                        const differenceCell = document.createElement('td');
                        differenceCell.className = `px-6 py-3 whitespace-nowrap text-sm ${scoreDiff > 0 ? 'text-green-600' : 'text-red-600'}`;
                        differenceCell.textContent = `${scoreDiff > 0 ? '+' : ''}${this.formatNumber(scoreDiff)}`;
                        iterationRow.appendChild(differenceCell);
                    } else {
                        const differenceCell = document.createElement('td');
                        differenceCell.className = 'px-6 py-3 whitespace-nowrap text-sm text-gray-500';
                        differenceCell.textContent = 'N/A';
                        iterationRow.appendChild(differenceCell);
                    }
                    
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
                    
                    meanRow.appendChild(createScoreCell(
                        selectedData.allFeatures.meanScore, 
                        null,
                        'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900'
                    ));
                    
                    meanRow.appendChild(createScoreCell(
                        selectedData.featureSubset ? selectedData.featureSubset.meanScore : null,
                        null,
                        'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900'
                    ));
                    
                    if (selectedData.featureSubset && 
                        selectedData.featureSubset.meanScore !== undefined &&
                        selectedData.allFeatures.meanScore !== undefined) {
                        const meanDiff = selectedData.featureSubset.meanScore - selectedData.allFeatures.meanScore;
                        const meanDiffCell = document.createElement('td');
                        meanDiffCell.className = `px-6 py-3 whitespace-nowrap text-sm font-medium ${
                            meanDiff > 0 ? 'text-green-600' : 'text-red-600'
                        }`;
                        meanDiffCell.textContent = `${meanDiff > 0 ? '+' : ''}${this.formatNumber(meanDiff)}`;
                        meanRow.appendChild(meanDiffCell);
                    } else {
                        const meanDiffCell = document.createElement('td');
                        meanDiffCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-500';
                        meanDiffCell.textContent = 'N/A';
                        meanRow.appendChild(meanDiffCell);
                    }
                    
                    tbody.appendChild(meanRow);
                }
                
                table.appendChild(tbody);
                tableContainer.appendChild(table);
                iterationsTab.appendChild(tableContainer);
                
                return iterationsTab;
                
            } catch (error) {
                console.error("Error creating iterations tab:", error);
                return this.createErrorMessage("Error creating iterations tab: " + error.message);
            }
        },
        
        // Create footer with try-catch
        createFooter: function() {
            try {
                const footer = document.createElement('div');
                footer.className = 'p-3 border-t border-gray-200 text-xs text-gray-500';
                
                const footerFlex = document.createElement('div');
                footerFlex.className = 'flex justify-between';
                
                // Left section
                const leftSpan = document.createElement('span');
                if (this.data) {
                    leftSpan.textContent = `Perturbation Test • ${this.data.modelType || 'Model'} • ${this.data.metric || 'Score'} Metric`;
                } else {
                    leftSpan.textContent = 'Perturbation Test';
                }
                footerFlex.appendChild(leftSpan);
                
                // Right section
                const rightSpan = document.createElement('span');
                if (this.data) {
                    rightSpan.textContent = `Base Score: ${this.data.baseScore ? this.formatNumber(this.data.baseScore) : 'N/A'} • Date: ${new Date().toLocaleDateString()}`;
                } else {
                    rightSpan.textContent = `Date: ${new Date().toLocaleDateString()}`;
                }
                footerFlex.appendChild(rightSpan);
                
                footer.appendChild(footerFlex);
                
                return footer;
                
            } catch (error) {
                console.error("Error creating footer:", error);
                const footer = document.createElement('div');
                footer.className = 'p-3 border-t border-gray-200 text-xs text-gray-500';
                footer.textContent = 'Error creating footer';
                return footer;
            }
        }
    };
    
    // Initialize when document is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Check if the container exists
        if (document.getElementById('perturbation-results-container')) {
            // Try to initialize with the safe controller
            try {
                window.SafePerturbationResultsController.init();
                console.log("Safe Perturbation Results Controller initialized");
            } catch (error) {
                console.error("Error initializing SafePerturbationResultsController:", error);
                
                // Create an error message in the container
                const container = document.getElementById('perturbation-results-container');
                if (container) {
                    container.innerHTML = `
                        <div class="p-4 text-center text-red-500 bg-red-50 rounded-lg">
                            <div class="text-2xl mb-2">❌</div>
                            <h3 class="font-bold">Error Loading Perturbation Results</h3>
                            <p class="mt-2">${error.message || 'An unknown error occurred'}</p>
                        </div>
                    `;
                }
            }
        }
    });
})();