// PerturbationResultsController.js
const PerturbationResultsController = {
    // Store the extracted data
    data: null,
    
    // Store UI state
    state: {
        selectedLevel: null,
        activeTab: 'summary',
        expandedSection: 'allFeatures'
    },
    
    /**
     * Initialize the controller
     */
    init: function() {
        console.log("Initializing Perturbation Results Controller");
        
        // Extract perturbation data if the manager exists
        if (typeof PerturbationResultsManager !== 'undefined') {
            this.data = PerturbationResultsManager.extractPerturbationData();
            
            // Set initial selected level to middle value if available, otherwise first value
            if (this.data.results.length > 0) {
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
    },
    
    /**
     * Render the perturbation results UI
     */
    render: function() {
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
    },
    
    /**
     * Render an error message
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
        });
    },
    
    /**
     * Create header section
     * @returns {HTMLElement} Header element
     */
    createHeader: function() {
        const header = document.createElement('div');
        header.className = 'p-4 border-b border-gray-200';
        
        // Title
        const title = document.createElement('h2');
        title.className = 'text-xl font-bold text-gray-800';
        title.textContent = 'Perturbation Test Results';
        header.appendChild(title);
        
        // Perturbation level selector
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
    },
    
    /**
     * Create summary tab content
     * @returns {HTMLElement} Summary tab element
     */
    createSummaryTab: function() {
        const selectedData = this.data.results.find(r => r.level === this.state.selectedLevel);
        if (!selectedData) {
            return this.createErrorMessage("No data available for selected level");
        }
        
        const summaryTab = document.createElement('div');
        summaryTab.className = 'p-4';
        
        // Feature summaries grid
        const summaryGrid = document.createElement('div');
        summaryGrid.className = 'grid grid-cols-1 md:grid-cols-2 gap-4';
        
        // All Features Summary
        summaryGrid.appendChild(this.createFeatureSummary(
            selectedData.allFeatures, 
            'allFeatures', 
            'All Features'
        ));
        
        // Feature Subset Summary
        summaryGrid.appendChild(this.createFeatureSummary(
            selectedData.featureSubset, 
            'featureSubset', 
            'Feature Subset'
        ));
        
        summaryTab.appendChild(summaryGrid);
        
        // Analysis box
        const analysisBox = document.createElement('div');
        analysisBox.className = 'mt-4 p-4 bg-blue-50 rounded-lg';
        
        const analysisTitle = document.createElement('h3');
        analysisTitle.className = 'font-medium text-blue-800 mb-2';
        analysisTitle.textContent = 'Analysis';
        analysisBox.appendChild(analysisTitle);
        
        const analysisParagraph = document.createElement('p');
        analysisParagraph.className = 'text-sm text-blue-700';
        
        // Generate analysis text
        let analysisText = `At ${this.state.selectedLevel * 100}% perturbation, the model shows `;
        if (selectedData.allFeatures.impact < 0) {
            analysisText += 'improvement ';
        } else {
            analysisText += `degradation of ${PerturbationResultsManager.formatNumber(selectedData.allFeatures.impact * 100, 2)}% `;
        }
        analysisText += 'when all features are perturbed. ';
        
        if (selectedData.featureSubset.impact < selectedData.allFeatures.impact) {
            analysisText += `The feature subset shows better robustness with only ${PerturbationResultsManager.formatNumber(selectedData.featureSubset.impact * 100, 2)}% impact.`;
        }
        
        analysisParagraph.textContent = analysisText;
        analysisBox.appendChild(analysisParagraph);
        
        summaryTab.appendChild(analysisBox);
        
        return summaryTab;
    },
    
    /**
     * Create feature summary panel
     * @param {Object} featureData - Data for feature type
     * @param {string} sectionId - Section identifier
     * @param {string} title - Section title
     * @returns {HTMLElement} Feature summary element
     */
    createFeatureSummary: function(featureData, sectionId, title) {
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
        impactSpan.className = `text-sm font-semibold ${PerturbationResultsManager.getImpactColorClass(featureData.impact)}`;
        impactSpan.textContent = `Impact: ${PerturbationResultsManager.formatNumber(featureData.impact * 100)}%`;
        panelHeader.appendChild(impactSpan);
        
        summaryPanel.appendChild(panelHeader);
        
        // Content
        const panelContent = document.createElement('div');
        panelContent.className = `p-4 ${this.state.expandedSection === sectionId ? 'block' : 'hidden'}`;
        
        // Stats table
        const statsTable = document.createElement('table');
        statsTable.className = 'min-w-full divide-y divide-gray-200';
        
        const tbody = document.createElement('tbody');
        tbody.className = 'divide-y divide-gray-200';
        
        // Base Score
        const baseScoreRow = document.createElement('tr');
        
        const baseScoreLabel = document.createElement('td');
        baseScoreLabel.className = 'px-3 py-2 text-sm font-medium text-gray-900';
        baseScoreLabel.textContent = 'Base Score';
        baseScoreRow.appendChild(baseScoreLabel);
        
        const baseScoreValue = document.createElement('td');
        baseScoreValue.className = 'px-3 py-2 text-sm text-gray-700';
        baseScoreValue.textContent = PerturbationResultsManager.formatNumber(featureData.baseScore);
        baseScoreRow.appendChild(baseScoreValue);
        
        tbody.appendChild(baseScoreRow);
        
        // Mean Score
        const meanScoreRow = document.createElement('tr');
        
        const meanScoreLabel = document.createElement('td');
        meanScoreLabel.className = 'px-3 py-2 text-sm font-medium text-gray-900';
        meanScoreLabel.textContent = 'Mean Score';
        meanScoreRow.appendChild(meanScoreLabel);
        
        const meanScoreValue = document.createElement('td');
        meanScoreValue.className = 'px-3 py-2 text-sm text-gray-700';
        
        const meanScoreText = document.createTextNode(
            PerturbationResultsManager.formatNumber(featureData.meanScore)
        );
        meanScoreValue.appendChild(meanScoreText);
        
        const meanScoreDiff = document.createElement('span');
        const diffValue = featureData.meanScore - featureData.baseScore;
        meanScoreDiff.className = `ml-2 ${diffValue >= 0 ? 'text-green-600' : 'text-red-600'}`;
        meanScoreDiff.textContent = `(${diffValue >= 0 ? '+' : ''}${PerturbationResultsManager.formatNumber(diffValue * 100, 2)}%)`;
        meanScoreValue.appendChild(meanScoreDiff);
        
        meanScoreRow.appendChild(meanScoreValue);
        tbody.appendChild(meanScoreRow);
        
        // Worst Score
        const worstScoreRow = document.createElement('tr');
        
        const worstScoreLabel = document.createElement('td');
        worstScoreLabel.className = 'px-3 py-2 text-sm font-medium text-gray-900';
        worstScoreLabel.textContent = 'Worst Score';
        worstScoreRow.appendChild(worstScoreLabel);
        
        const worstScoreValue = document.createElement('td');
        worstScoreValue.className = 'px-3 py-2 text-sm text-red-600';
        
        const worstScoreText = document.createTextNode(
            PerturbationResultsManager.formatNumber(featureData.worstScore)
        );
        worstScoreValue.appendChild(worstScoreText);
        
        const worstScoreDiff = document.createElement('span');
        worstScoreDiff.className = 'ml-2';
        worstScoreDiff.textContent = `(${PerturbationResultsManager.formatNumber((featureData.worstScore - featureData.baseScore) * 100, 2)}%)`;
        worstScoreValue.appendChild(worstScoreDiff);
        
        worstScoreRow.appendChild(worstScoreValue);
        tbody.appendChild(worstScoreRow);
        
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
    },
    
    /**
     * Create iterations tab content
     * @returns {HTMLElement} Iterations tab element
     */
    createIterationsTab: function() {
        const selectedData = this.data.results.find(r => r.level === this.state.selectedLevel);
        if (!selectedData) {
            return this.createErrorMessage("No data available for selected level");
        }
        
        const iterationsTab = document.createElement('div');
        iterationsTab.className = 'p-4';
        
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
        baseAllFeaturesCell.textContent = PerturbationResultsManager.formatNumber(selectedData.allFeatures.baseScore);
        baseRow.appendChild(baseAllFeaturesCell);
        
        const baseFeatureSubsetCell = document.createElement('td');
        baseFeatureSubsetCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        baseFeatureSubsetCell.textContent = PerturbationResultsManager.formatNumber(selectedData.featureSubset.baseScore);
        baseRow.appendChild(baseFeatureSubsetCell);
        
        const baseDifferenceCell = document.createElement('td');
        baseDifferenceCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        baseDifferenceCell.textContent = '0.0000';
        baseRow.appendChild(baseDifferenceCell);
        
        tbody.appendChild(baseRow);
        
        // Iteration rows
        const allFeaturesIterations = selectedData.allFeatures.iterations || [];
        const featureSubsetIterations = selectedData.featureSubset.iterations || [];
        
        const maxIterations = Math.max(allFeaturesIterations.length, featureSubsetIterations.length);
        
        for (let i = 0; i < maxIterations; i++) {
            const iterationRow = document.createElement('tr');
            
            // Iteration number
            const iterationCell = document.createElement('td');
            iterationCell.className = 'px-6 py-3 whitespace-nowrap text-sm text-gray-900';
            iterationCell.textContent = `#${i + 1}`;
            iterationRow.appendChild(iterationCell);
            
            // All features score
            const allFeaturesScore = allFeaturesIterations[i] || selectedData.allFeatures.meanScore;
            const allFeaturesCell = document.createElement('td');
            allFeaturesCell.className = `px-6 py-3 whitespace-nowrap text-sm text-gray-700 ${
                PerturbationResultsManager.getScoreBgColorClass(allFeaturesScore, selectedData.allFeatures.baseScore)
            }`;
            
            const allFeaturesScoreText = document.createTextNode(
                PerturbationResultsManager.formatNumber(allFeaturesScore)
            );
            allFeaturesCell.appendChild(allFeaturesScoreText);
            
            const allFeaturesDiff = allFeaturesScore - selectedData.allFeatures.baseScore;
            const allFeaturesDiffSpan = document.createElement('span');
            allFeaturesDiffSpan.className = `ml-2 text-xs ${allFeaturesDiff >= 0 ? 'text-green-600' : 'text-red-600'}`;
            allFeaturesDiffSpan.textContent = `(${allFeaturesDiff >= 0 ? '+' : ''}${PerturbationResultsManager.formatNumber(allFeaturesDiff * 100, 2)}%)`;
            allFeaturesCell.appendChild(allFeaturesDiffSpan);
            
            iterationRow.appendChild(allFeaturesCell);
            
            // Feature subset score
            const featureSubsetScore = featureSubsetIterations[i] || selectedData.featureSubset.meanScore;
            const featureSubsetCell = document.createElement('td');
            featureSubsetCell.className = `px-6 py-3 whitespace-nowrap text-sm text-gray-700 ${
                PerturbationResultsManager.getScoreBgColorClass(featureSubsetScore, selectedData.featureSubset.baseScore)
            }`;
            
            const featureSubsetScoreText = document.createTextNode(
                PerturbationResultsManager.formatNumber(featureSubsetScore)
            );
            featureSubsetCell.appendChild(featureSubsetScoreText);
            
            const featureSubsetDiff = featureSubsetScore - selectedData.featureSubset.baseScore;
            const featureSubsetDiffSpan = document.createElement('span');
            featureSubsetDiffSpan.className = `ml-2 text-xs ${featureSubsetDiff >= 0 ? 'text-green-600' : 'text-red-600'}`;
            featureSubsetDiffSpan.textContent = `(${featureSubsetDiff >= 0 ? '+' : ''}${PerturbationResultsManager.formatNumber(featureSubsetDiff * 100, 2)}%)`;
            featureSubsetCell.appendChild(featureSubsetDiffSpan);
            
            iterationRow.appendChild(featureSubsetCell);
            
            // Difference between scores
            const scoreDiff = featureSubsetScore - allFeaturesScore;
            const differenceCell = document.createElement('td');
            differenceCell.className = `px-6 py-3 whitespace-nowrap text-sm ${scoreDiff > 0 ? 'text-green-600' : 'text-red-600'}`;
            differenceCell.textContent = `${scoreDiff > 0 ? '+' : ''}${PerturbationResultsManager.formatNumber(scoreDiff)}`;
            
            iterationRow.appendChild(differenceCell);
            
            tbody.appendChild(iterationRow);
        }
        
        // Mean row
        const meanRow = document.createElement('tr');
        meanRow.className = 'bg-gray-100';
        
        const meanLabelCell = document.createElement('td');
        meanLabelCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-bold text-gray-900';
        meanLabelCell.textContent = 'Mean';
        meanRow.appendChild(meanLabelCell);
        
        const allFeaturesMeanCell = document.createElement('td');
        allFeaturesMeanCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        allFeaturesMeanCell.textContent = PerturbationResultsManager.formatNumber(selectedData.allFeatures.meanScore);
        meanRow.appendChild(allFeaturesMeanCell);
        
        const featureSubsetMeanCell = document.createElement('td');
        featureSubsetMeanCell.className = 'px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900';
        featureSubsetMeanCell.textContent = PerturbationResultsManager.formatNumber(selectedData.featureSubset.meanScore);
        meanRow.appendChild(featureSubsetMeanCell);
        
        const meanDiffCell = document.createElement('td');
        const meanDiff = selectedData.featureSubset.meanScore - selectedData.allFeatures.meanScore;
        meanDiffCell.className = `px-6 py-3 whitespace-nowrap text-sm font-medium ${
            meanDiff > 0 ? 'text-green-600' : 'text-red-600'
        }`;
        meanDiffCell.textContent = `${meanDiff > 0 ? '+' : ''}${PerturbationResultsManager.formatNumber(meanDiff)}`;
        meanRow.appendChild(meanDiffCell);
        
        tbody.appendChild(meanRow);
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        iterationsTab.appendChild(tableContainer);
        
        // Additional stats
        const statsGrid = document.createElement('div');
        statsGrid.className = 'mt-4 grid grid-cols-1 md:grid-cols-2 gap-4';
        
        // Worst performance
        const worstPerformanceDiv = document.createElement('div');
        worstPerformanceDiv.className = 'p-3 bg-yellow-50 rounded-lg';
        
        const worstPerfTitle = document.createElement('h3');
        worstPerfTitle.className = 'font-medium text-yellow-800 mb-1';
        worstPerfTitle.textContent = 'Worst Performance';
        worstPerformanceDiv.appendChild(worstPerfTitle);
        
        const allFeaturesWorst = document.createElement('p');
        allFeaturesWorst.className = 'text-sm text-yellow-700';
        allFeaturesWorst.innerHTML = `All Features: ${PerturbationResultsManager.formatNumber(selectedData.allFeatures.worstScore)} 
            <span class="ml-1">
                (${PerturbationResultsManager.formatNumber((selectedData.allFeatures.worstScore - selectedData.allFeatures.baseScore) * 100, 2)}%)
            </span>`;
        worstPerformanceDiv.appendChild(allFeaturesWorst);
        
        const featureSubsetWorst = document.createElement('p');
        featureSubsetWorst.className = 'text-sm text-yellow-700';
        featureSubsetWorst.innerHTML = `Feature Subset: ${PerturbationResultsManager.formatNumber(selectedData.featureSubset.worstScore)}
            <span class="ml-1">
                (${PerturbationResultsManager.formatNumber((selectedData.featureSubset.worstScore - selectedData.featureSubset.baseScore) * 100, 2)}%)
            </span>`;
        worstPerformanceDiv.appendChild(featureSubsetWorst);
        
        statsGrid.appendChild(worstPerformanceDiv);
        
        // Standard deviation
        const stdDevDiv = document.createElement('div');
        stdDevDiv.className = 'p-3 bg-blue-50 rounded-lg';
        
        const stdDevTitle = document.createElement('h3');
        stdDevTitle.className = 'font-medium text-blue-800 mb-1';
        stdDevTitle.textContent = 'Standard Deviation';
        stdDevDiv.appendChild(stdDevTitle);
        
        // Calculate standard deviations
        const calculateStdDev = (scores, mean) => {
            if (!scores || scores.length === 0) return 0;
            const squaredDiffs = scores.map(score => Math.pow(score - mean, 2));
            const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / scores.length;
            return Math.sqrt(variance);
        };
        
        const allFeaturesStdDev = calculateStdDev(
            selectedData.allFeatures.iterations,
            selectedData.allFeatures.meanScore
        );
        
        const featureSubsetStdDev = calculateStdDev(
            selectedData.featureSubset.iterations,
            selectedData.featureSubset.meanScore
        );
        
        const allFeaturesStdDevP = document.createElement('p');
        allFeaturesStdDevP.className = 'text-sm text-blue-700';
        allFeaturesStdDevP.textContent = `All Features: ${PerturbationResultsManager.formatNumber(allFeaturesStdDev)}`;
        stdDevDiv.appendChild(allFeaturesStdDevP);
        
        const featureSubsetStdDevP = document.createElement('p');
        featureSubsetStdDevP.className = 'text-sm text-blue-700';
        featureSubsetStdDevP.textContent = `Feature Subset: ${PerturbationResultsManager.formatNumber(featureSubsetStdDev)}`;
        stdDevDiv.appendChild(featureSubsetStdDevP);
        
        statsGrid.appendChild(stdDevDiv);
        iterationsTab.appendChild(statsGrid);
        
        return iterationsTab;
    },
    
    /**
     * Create footer section
     * @returns {HTMLElement} Footer element
     */
    createFooter: function() {
        const footer = document.createElement('div');
        footer.className = 'p-3 border-t border-gray-200 text-xs text-gray-500';
        
        const footerFlex = document.createElement('div');
        footerFlex.className = 'flex justify-between';
        
        const leftSpan = document.createElement('span');
        leftSpan.textContent = `Perturbation Test • ${this.data.modelType} • ${this.data.metric} Metric`;
        footerFlex.appendChild(leftSpan);
        
        const rightSpan = document.createElement('span');
        rightSpan.textContent = `Base Score: ${PerturbationResultsManager.formatNumber(this.data.baseScore)} • Date: ${new Date().toLocaleDateString()}`;
        footerFlex.appendChild(rightSpan);
        
        footer.appendChild(footerFlex);
        
        return footer;
    },
    
    /**
     * Create error message
     * @param {string} message - Error message to display
     * @returns {HTMLElement} Error message element
     */
    createErrorMessage: function(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'p-8 text-center text-red-500';
        errorDiv.textContent = message || 'An error occurred loading perturbation data';
        return errorDiv;
    }
};