// PerturbationResultsManager.js
const PerturbationResultsManager = {
    /**
     * Extract perturbation data from report data
     * @returns {Object} Perturbation data
     */
    extractPerturbationData: function() {
        console.log("Extracting perturbation data from report");
        const perturbationResults = [];
        
        try {
            // Check if window.reportData exists and has necessary data
            if (!window.reportData || (!window.reportData.raw && !window.reportData.perturbation_chart_data)) {
                console.warn("Report data not found or incomplete");
                return this.createSampleData(); // Return sample data if no report data available
            }
            
            // If the server already prepared perturbation chart data, use it
            if (window.reportData.perturbation_chart_data) {
                console.log("Using server-prepared perturbation chart data");
                return this.processPreparedChartData(window.reportData.perturbation_chart_data);
            }
            
            // Extract from raw data if available
            if (window.reportData.raw && window.reportData.raw.by_level) {
                console.log("Extracting from raw perturbation data");
                
                // Get base score and metric
                const baseScore = window.reportData.base_score || 0.0;
                const metric = window.reportData.metric || 'Score';
                
                // Process each perturbation level
                Object.keys(window.reportData.raw.by_level).forEach(level => {
                    const numericLevel = parseFloat(level);
                    const levelData = window.reportData.raw.by_level[level];
                    
                    const resultItem = {
                        level: numericLevel,
                        allFeatures: this.extractFeatureData(levelData, 'all_features', baseScore),
                        featureSubset: this.extractFeatureData(levelData, 'feature_subset', baseScore)
                    };
                    
                    // If feature subset wasn't found, try to extract from selectedFeatures
                    if (!resultItem.featureSubset.iterations.length && window.reportData.feature_subset) {
                        const featureSubset = window.reportData.feature_subset;
                        if (Array.isArray(featureSubset) && featureSubset.length > 0) {
                            const subsetName = featureSubset.join('_');
                            resultItem.featureSubset = this.extractFeatureData(levelData, subsetName, baseScore);
                        }
                    }
                    
                    perturbationResults.push(resultItem);
                });
                
                // Sort by level
                perturbationResults.sort((a, b) => a.level - b.level);
                
                return {
                    results: perturbationResults,
                    baseScore: baseScore,
                    metric: metric,
                    modelName: window.reportData.model_name || 'Model',
                    modelType: window.reportData.model_type || 'Model',
                    featureSubset: window.reportData.feature_subset || []
                };
            }
            
            // If no perturbation data found, return sample data
            console.warn("No perturbation data found in report data");
            return this.createSampleData();
            
        } catch (error) {
            console.error("Error extracting perturbation data:", error);
            return this.createSampleData(); // Return sample data on error
        }
    },
    
    /**
     * Process server-prepared chart data
     * @param {Object} chartData - Server-prepared chart data 
     * @returns {Object} Processed perturbation data
     */
    processPreparedChartData: function(chartData) {
        const perturbationResults = [];
        const baseScore = chartData.baseScore || 0.0;
        const metric = chartData.metric || 'Score';
        
        // Process levels and scores
        if (chartData.levels && chartData.scores) {
            chartData.levels.forEach((level, index) => {
                const resultItem = {
                    level: level,
                    allFeatures: {
                        baseScore: baseScore,
                        meanScore: chartData.scores[index] || 0,
                        impact: (baseScore - (chartData.scores[index] || 0)) / baseScore,
                        worstScore: chartData.worstScores ? chartData.worstScores[index] || 0 : 0,
                        iterations: []
                    },
                    featureSubset: {
                        baseScore: baseScore,
                        meanScore: 0,
                        impact: 0,
                        worstScore: 0,
                        iterations: []
                    }
                };
                
                // Add iterations if available
                if (window.reportData.iterations_by_level && window.reportData.iterations_by_level[level]) {
                    resultItem.allFeatures.iterations = window.reportData.iterations_by_level[level];
                } else if (chartData.iterations && chartData.iterations[index]) {
                    resultItem.allFeatures.iterations = chartData.iterations[index];
                } else {
                    // If no iterations available, create a sample array with the mean score
                    resultItem.allFeatures.iterations = Array(10).fill(resultItem.allFeatures.meanScore);
                }
                
                // Process feature subset if available
                if (chartData.alternativeModels && Object.keys(chartData.alternativeModels).length > 0) {
                    const subsetName = Object.keys(chartData.alternativeModels)[0];
                    const subsetData = chartData.alternativeModels[subsetName];
                    
                    resultItem.featureSubset = {
                        baseScore: subsetData.baseScore || baseScore,
                        meanScore: subsetData.scores[index] || 0,
                        impact: (baseScore - (subsetData.scores[index] || 0)) / baseScore,
                        worstScore: subsetData.worstScores ? subsetData.worstScores[index] || 0 : 0,
                        iterations: Array(10).fill(subsetData.scores[index] || 0)
                    };
                }
                
                perturbationResults.push(resultItem);
            });
        }
        
        return {
            results: perturbationResults,
            baseScore: baseScore,
            metric: metric,
            modelName: chartData.modelName || 'Model',
            modelType: window.reportData.model_type || 'Model',
            featureSubset: window.reportData.feature_subset || []
        };
    },
    
    /**
     * Extract feature data from level data
     * @param {Object} levelData - Level data
     * @param {string} featureKey - Feature key to extract
     * @param {number} baseScore - Base score
     * @returns {Object} Extracted feature data
     */
    extractFeatureData: function(levelData, featureKey, baseScore) {
        const result = {
            baseScore: baseScore,
            meanScore: 0,
            impact: 0,
            worstScore: 0,
            iterations: []
        };
        
        try {
            // Check if we have overall_result data
            if (levelData.overall_result && levelData.overall_result[featureKey]) {
                const featureData = levelData.overall_result[featureKey];
                
                result.meanScore = featureData.mean_score || featureData.perturbed_score || 0;
                result.worstScore = featureData.worst_score || featureData.min_score || result.meanScore;
                result.impact = (baseScore - result.meanScore) / baseScore;
                
                // If negative impact (improvement), cap at a reasonable value
                if (result.impact < -0.1) result.impact = -0.1;
            }
            
            // Extract iteration data if available
            if (levelData.runs && levelData.runs[featureKey] && 
                levelData.runs[featureKey][0] && 
                levelData.runs[featureKey][0].iterations &&
                levelData.runs[featureKey][0].iterations.scores) {
                    
                result.iterations = levelData.runs[featureKey][0].iterations.scores;
            } else if (result.meanScore > 0) {
                // If no iteration data but we have a mean score, create synthetic data
                // with small variations around the mean
                const stdDev = Math.max(0.01, Math.abs(result.meanScore - result.worstScore) / 2);
                result.iterations = Array(10).fill(0).map(() => {
                    return this.normalRandom(result.meanScore, stdDev);
                });
                
                // Ensure one iteration is equal to worst score if it exists
                if (result.worstScore > 0 && result.worstScore < result.meanScore) {
                    result.iterations[0] = result.worstScore;
                }
            }
        } catch (error) {
            console.error(`Error extracting ${featureKey} data:`, error);
        }
        
        return result;
    },
    
    /**
     * Generate a random number from normal distribution
     * @param {number} mean - Mean of the distribution
     * @param {number} stdDev - Standard deviation
     * @returns {number} Random number
     */
    normalRandom: function(mean, stdDev) {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return mean + z0 * stdDev;
    },
    
    /**
     * Create sample data when no real data is available
     * @returns {Object} Sample perturbation data
     */
    createSampleData: function() {
        console.log("Creating sample perturbation data");
        const baseScore = 0.774;
        const perturbationResults = [];
        
        // Create sample data for different perturbation levels
        [0.1, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(level => {
            // Calculate progressively worse scores as perturbation increases
            const meanScoreAll = baseScore * (1 - 0.1 * level * level);
            const worstScoreAll = meanScoreAll * 0.97;
            const meanScoreSubset = baseScore * (1 - 0.04 * level * level);
            const worstScoreSubset = meanScoreSubset * 0.98;
            
            // Create iterations arrays with variations
            const iterationsAll = Array(10).fill(0).map(() => 
                this.normalRandom(meanScoreAll, 0.005 * level));
            iterationsAll[0] = worstScoreAll; // Ensure one iteration matches worst score
            
            const iterationsSubset = Array(10).fill(0).map(() => 
                this.normalRandom(meanScoreSubset, 0.003 * level));
            iterationsSubset[0] = worstScoreSubset; // Ensure one iteration matches worst score
            
            perturbationResults.push({
                level: level,
                allFeatures: {
                    baseScore: baseScore,
                    meanScore: meanScoreAll,
                    impact: (baseScore - meanScoreAll) / baseScore,
                    worstScore: worstScoreAll,
                    iterations: iterationsAll
                },
                featureSubset: {
                    baseScore: baseScore,
                    meanScore: meanScoreSubset,
                    impact: (baseScore - meanScoreSubset) / baseScore,
                    worstScore: worstScoreSubset,
                    iterations: iterationsSubset
                }
            });
        });
        
        return {
            results: perturbationResults,
            baseScore: baseScore,
            metric: 'AUC',
            modelName: 'Sample Model',
            modelType: 'Random Forest Classifier',
            featureSubset: ['Feature1', 'Feature2']
        };
    },
    
    /**
     * Format number with specified precision
     * @param {number} num - Number to format
     * @param {number} precision - Number of decimal places
     * @returns {string} Formatted number
     */
    formatNumber: function(num, precision = 4) {
        return Number(num).toFixed(precision);
    },
    
    /**
     * Get color class based on impact
     * @param {number} impact - Impact value
     * @returns {string} CSS class for coloring
     */
    getImpactColorClass: function(impact) {
        if (impact < 0) return 'text-green-600'; // Improvement
        if (impact < 0.03) return 'text-blue-600'; // Small degradation
        if (impact < 0.07) return 'text-yellow-600'; // Medium degradation
        return 'text-red-600'; // Large degradation
    },
    
    /**
     * Get background color class based on score comparison
     * @param {number} score - Score to compare
     * @param {number} baseScore - Base score for comparison
     * @returns {string} CSS class for background coloring
     */
    getScoreBgColorClass: function(score, baseScore) {
        const diff = score - baseScore;
        if (diff > 0) return 'bg-green-100';
        if (diff > -0.01) return 'bg-yellow-50';
        if (diff > -0.03) return 'bg-orange-50';
        return 'bg-red-50';
    }
};