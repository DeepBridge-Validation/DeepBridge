/**
 * PerturbationResultsManager
 * Responsible for extracting and processing perturbation test data
 * to be displayed in the details tab of robustness reports
 */
window.PerturbationResultsManager = {
    /**
     * Extract perturbation data from report data
     * @returns {Object} Object containing processed perturbation data
     */
    extractPerturbationData: function() {
        console.log("Extracting perturbation data");
        
        try {
            // Check if report data is available
            if (!window.reportData) {
                console.error("No report data available");
                return null;
            }
            
            // Basic result structure
            const result = {
                modelName: window.reportData.model_name || 'Model',
                modelType: window.reportData.model_type || 'Unknown',
                metric: window.reportData.metric || 'Score',
                baseScore: window.reportData.base_score || 0,
                results: []
            };
            
            // Extract results from raw data if available
            if (window.reportData.raw && window.reportData.raw.by_level) {
                const byLevel = window.reportData.raw.by_level;
                
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
                        allFeatures: this._extractFeatureData(levelData, 'all_features', result.baseScore),
                        featureSubset: this._extractFeatureData(levelData, 'feature_subset', result.baseScore)
                    };
                    
                    // Add iteration scores from runs data if available
                    if (levelData.runs) {
                        if (levelData.runs.all_features && levelData.runs.all_features.length > 0) {
                            levelResult.allFeatures.iterations = this._extractIterationScores(levelData.runs.all_features);
                        }
                        
                        if (levelData.runs.feature_subset && levelData.runs.feature_subset.length > 0) {
                            levelResult.featureSubset.iterations = this._extractIterationScores(levelData.runs.feature_subset);
                        }
                    }
                    
                    result.results.push(levelResult);
                });
            } else {
                console.warn("No raw perturbation data found");
            }
            
            return result;
        } catch (error) {
            console.error("Error extracting perturbation data:", error);
            return null;
        }
    },
    
    /**
     * Extract feature data from level results
     * @param {Object} levelData - Level data from raw results
     * @param {string} featureType - Feature type (all_features or feature_subset)
     * @param {number} baseScore - Base score for comparison
     * @returns {Object} Processed feature data
     */
    _extractFeatureData: function(levelData, featureType, baseScore) {
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
    },
    
    /**
     * Extract iteration scores from runs data
     * @param {Array} runsData - Array of run results
     * @returns {Array} Array of iteration scores
     */
    _extractIterationScores: function(runsData) {
        const iterationScores = [];
        
        runsData.forEach(run => {
            if (run.iterations && run.iterations.scores && Array.isArray(run.iterations.scores)) {
                // Add all scores from this run to the result
                iterationScores.push(...run.iterations.scores);
            }
        });
        
        return iterationScores;
    },
    
    /**
     * Format a number for display
     * @param {number} value - Number to format
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted number
     */
    formatNumber: function(value, decimals = 4) {
        if (value === undefined || value === null || isNaN(value)) {
            return 'N/A';
        }
        return value.toFixed(decimals);
    },
    
    /**
     * Get color class for impact value
     * @param {number} impact - Impact value
     * @returns {string} CSS class
     */
    getImpactColorClass: function(impact) {
        if (impact < 0) return 'text-green-600';
        if (impact < 0.05) return 'text-yellow-600';
        if (impact < 0.1) return 'text-orange-600';
        return 'text-red-600';
    },
    
    /**
     * Get background color class for score comparison
     * @param {number} score - Score to compare
     * @param {number} baseScore - Base score
     * @returns {string} CSS class
     */
    getScoreBgColorClass: function(score, baseScore) {
        if (!score || !baseScore) return '';
        const diff = score - baseScore;
        if (diff > 0) return 'bg-green-50';
        if (diff < -0.1) return 'bg-red-50';
        if (diff < -0.05) return 'bg-orange-50';
        if (diff < 0) return 'bg-yellow-50';
        return '';
    }
};