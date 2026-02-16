/**
 * Model comparison chart fix
 * Direct replacement of the problematic functions
 * Updated: May 7, 2024 - Added fix for "Details by Level" chart
 */

// Define a safe version of the extractModelComparisonData function
window.SafeModelComparisonChartManager = {
    /**
     * Extract model comparison data from report data
     * Safe version with proper error handling - NO synthetic data generation
     * @returns {Object} Data for model comparison chart
     */
    extractModelComparisonData: function() {
        console.log("Using safe model comparison data extractor v2.0 (No Synthetic Data)");
        
        // Default safe return object with no trailing commas
        const safeReturnObject = {
            levels: [],
            modelScores: {},
            modelNames: {},
            metricName: ""
        };
        
        try {
            // Check if we have valid report data
            if (!window.reportData) {
                console.warn("No reportData available");
                return safeReturnObject;
            }
            
            // Check for alternative models
            const hasAlternativeModels = window.reportData.alternative_models && 
                                        Object.keys(window.reportData.alternative_models).length > 0;
            
            if (!hasAlternativeModels) {
                console.warn("No alternative models found - won't create synthetic models");
                return safeReturnObject;
            }
            
            // Get primary model name
            const primaryModelName = window.reportData.model_name || "Primary Model";
            const metricName = window.reportData.metric || "Score";
            
            // Determine available levels
            const rawData = window.reportData.raw && window.reportData.raw.by_level || {};
            const perturbationData = window.reportData.perturbation_chart_data || {};
            
            // Use perturbation chart data levels if available, otherwise extract from raw data
            let levels = [];
            if (perturbationData && perturbationData.levels) {
                levels = perturbationData.levels;
                console.log("Using levels from perturbation chart data:", levels);
            } else {
                // Extract levels from raw data
                levels = Object.keys(rawData)
                    .map(level => parseFloat(level))
                    .filter(level => !isNaN(level))
                    .sort((a, b) => a - b);
                console.log("Extracted levels from raw data:", levels);
            }
            
            if (levels.length === 0) {
                console.warn("No valid perturbation levels found");
                return safeReturnObject;
            }
            
            // Initialize model scores and names
            const modelScores = {};
            const modelNames = {};
            
            // Add primary model if we have perturbation chart data
            if (perturbationData.scores && perturbationData.scores.length > 0) {
                modelScores["primary"] = perturbationData.scores;
                modelNames["primary"] = primaryModelName;
                console.log("Added primary model scores from perturbation_chart_data");
            } 
            // Otherwise extract from raw data
            else if (Object.keys(rawData).length > 0) {
                const primaryScores = levels.map(level => {
                    const levelStr = level.toString();
                    if (rawData[levelStr] && 
                        rawData[levelStr].overall_result && 
                        rawData[levelStr].overall_result.all_features) {
                        const score = rawData[levelStr].overall_result.all_features.mean_score;
                        return typeof score === 'number' ? score : null;
                    }
                    return null;
                });
                
                if (primaryScores.some(score => score !== null)) {
                    modelScores["primary"] = primaryScores;
                    modelNames["primary"] = primaryModelName;
                    console.log("Added primary model scores from raw data");
                }
            }
            
            // Process alternative models - ONLY use real data, no synthetic score generation
            const alternativeModels = window.reportData.alternative_models || {};
            
            // Count how many alternative models have real perturbation test data
            let modelsWithRealData = 0;
            
            Object.entries(alternativeModels).forEach(([modelId, modelData]) => {
                // Get model name
                const name = modelData.model_name || modelId;
                
                // Only include models that have actual test data for perturbation levels
                if (modelData.perturbation_results && 
                    Object.keys(modelData.perturbation_results).length > 0) {
                    
                    // Extract actual scores for each perturbation level
                    const scores = levels.map(level => {
                        const levelStr = level.toString();
                        if (modelData.perturbation_results[levelStr] && 
                            modelData.perturbation_results[levelStr].overall_result) {
                            return modelData.perturbation_results[levelStr].overall_result.mean_score || null;
                        }
                        return null;
                    });
                    
                    // Only include models with at least one valid score
                    if (scores.some(score => score !== null)) {
                        modelScores[modelId] = scores;
                        modelNames[modelId] = name;
                        modelsWithRealData++;
                        console.log(`Added real data for alternative model: ${name}`);
                    }
                }
            });
            
            // Log warning if no models have real perturbation test data
            if (modelsWithRealData === 0) {
                console.warn("No alternative models with real perturbation test data found");
                
                // Show error message on the page if the element exists
                const chartContainer = document.getElementById('modelComparisonChartContainer');
                if (chartContainer) {
                    chartContainer.innerHTML = `
                        <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                            <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                            <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados de perturbação não disponíveis</h3>
                            <p style="color: #333; font-size: 16px; line-height: 1.4;">Os modelos alternativos não possuem dados de testes de perturbação.</p>
                            <p style="color: #333; margin-top: 20px; font-size: 14px;">Execute testes de robustez em todos os modelos para comparação.</p>
                        </div>`;
                }
            }
            
            // Log final data
            console.log("Final data for model comparison chart:");
            console.log("- Levels:", levels);
            console.log("- Models:", Object.keys(modelScores));
            
            // Return the processed data with NO TRAILING COMMAS
            return {
                levels: levels,
                modelScores: modelScores,
                modelNames: modelNames,
                metricName: metricName
            };
        } catch (error) {
            console.error("Error extracting model comparison data:", error);
            return safeReturnObject;
        }
    },
    
    /**
     * Fixed version of extractModelLevelDetailsData function
     * Properly extracts data for Model Level Details chart (no synthetic data)
     */
    extractModelLevelDetailsData: function() {
        console.log("Using safe extractModelLevelDetailsData (no synthetic data)");
        
        // Default safe return object
        const safeReturnObject = {
            levels: [],
            modelScores: {},
            modelNames: {},
            metricName: "Score"
        };
        
        try {
            if (!window.reportData) {
                console.warn("No reportData available");
                return safeReturnObject;
            }
            
            // Get the metric name from reportData
            let metricName = "Score";
            if (window.reportData.metric) {
                metricName = window.reportData.metric;
            } else if (window.reportConfig && window.reportConfig.metric) {
                metricName = window.reportConfig.metric;
            }
            
            // Determine which data path to use based on available structure
            let primaryModelData = null;
            let alternativeModelsData = null;
            let rawDataPath = null;
            
            // Check new data structure first (results.robustness.primary_model)
            if (window.reportData.results && 
                window.reportData.results.robustness && 
                window.reportData.results.robustness.primary_model) {
                console.log("Using new data structure (results.robustness.primary_model)");
                primaryModelData = window.reportData.results.robustness.primary_model;
                
                if (window.reportData.results.robustness.alternative_models) {
                    alternativeModelsData = window.reportData.results.robustness.alternative_models;
                }
                
                if (primaryModelData.raw && primaryModelData.raw.by_level) {
                    rawDataPath = primaryModelData.raw.by_level;
                }
            } 
            // Fallback to original structure
            else if (window.reportData.raw && window.reportData.raw.by_level) {
                console.log("Using original data structure (raw.by_level)");
                rawDataPath = window.reportData.raw.by_level;
                
                if (window.reportData.alternative_models) {
                    alternativeModelsData = window.reportData.alternative_models;
                }
            }
            
            if (!rawDataPath) {
                console.warn("No valid raw data path found");
                return safeReturnObject;
            }
            
            // Get perturbation levels from the data
            const levels = Object.keys(rawDataPath)
                .map(level => parseFloat(level))
                .filter(level => !isNaN(level))
                .sort((a, b) => a - b);
                
            if (levels.length === 0) {
                console.warn("No valid levels found");
                return safeReturnObject;
            }
            
            console.log("Found levels:", levels);
            
            // Get primary model name
            const primaryModelName = window.reportData.model_name || "Primary Model";
            
            // Initialize result data structures
            const modelScores = {};
            const modelNames = {};
            
            // Extract primary model scores
            const primaryScores = levels.map(level => {
                const levelStr = level.toString();
                if (rawDataPath[levelStr]) {
                    // Try different potential paths where scores might be stored
                    const paths = [
                        // Try specific path from user data
                        rawDataPath[levelStr].runs?.all_features?.[0]?.perturbed_score,
                        // Standard path for mean scores
                        rawDataPath[levelStr].overall_result?.all_features?.mean_score,
                        // Other potential paths
                        rawDataPath[levelStr].mean_score,
                        rawDataPath[levelStr].perturbed_score,
                        rawDataPath[levelStr].results?.overall_result?.all_features?.mean_score
                    ];
                    
                    // Use the first valid score from any path
                    for (const path of paths) {
                        if (typeof path === 'number') {
                            return path;
                        }
                    }
                }
                return null;
            });
            
            if (primaryScores.some(score => score !== null)) {
                modelScores["primary"] = primaryScores;
                modelNames["primary"] = primaryModelName;
                console.log("Added primary model scores");
            }
            
            // Extract alternative model scores
            if (alternativeModelsData && Object.keys(alternativeModelsData).length > 0) {
                console.log("Processing alternative models:", Object.keys(alternativeModelsData));
                
                Object.entries(alternativeModelsData).forEach(([modelId, modelData]) => {
                    // Skip if no raw data
                    if (!modelData.raw || !modelData.raw.by_level) {
                        console.log(`Model ${modelId} has no raw data`);
                        return;
                    }
                    
                    // Get model name
                    const name = modelData.model_name || modelId;
                    
                    // Extract scores for each level
                    const scores = levels.map(level => {
                        const levelStr = level.toString();
                        if (!modelData.raw.by_level[levelStr]) {
                            return null;
                        }
                        
                        const paths = [
                            // Try specific path from user data
                            modelData.raw.by_level[levelStr].runs?.all_features?.[0]?.perturbed_score,
                            // Standard path for mean scores
                            modelData.raw.by_level[levelStr].overall_result?.all_features?.mean_score,
                            // Other potential paths
                            modelData.raw.by_level[levelStr].mean_score,
                            modelData.raw.by_level[levelStr].perturbed_score,
                            modelData.raw.by_level[levelStr].results?.overall_result?.all_features?.mean_score
                        ];
                        
                        // Use the first valid score
                        for (const path of paths) {
                            if (typeof path === 'number') {
                                return path;
                            }
                        }
                        
                        return null;
                    });
                    
                    // Only add if we found at least one valid score
                    if (scores.some(score => score !== null)) {
                        modelScores[modelId] = scores;
                        modelNames[modelId] = name;
                        console.log(`Added scores for model ${name}`);
                    } else {
                        console.log(`No valid scores found for model ${name}`);
                    }
                });
            }
            
            // Check if we have at least two models with data
            const validModels = Object.keys(modelScores);
            if (validModels.length < 2) {
                console.warn("Not enough models for comparison (need at least 2)");
                console.log("Models with valid data:", validModels);
                return safeReturnObject;
            }
            
            return {
                levels: levels,
                modelScores: modelScores,
                modelNames: modelNames,
                metricName: metricName
            };
        } catch (error) {
            console.error("Error extracting model level details data:", error);
            return safeReturnObject;
        }
    }
};

// Set up the fix to be applied when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Replace the problematic functions with our safe versions
    if (window.ModelComparisonChartManager) {
        console.log("Replacing ModelComparisonChartManager.extractModelComparisonData with safe version");
        window.ModelComparisonChartManager.extractModelComparisonData = window.SafeModelComparisonChartManager.extractModelComparisonData;
    }
    
    // Fix the "Details by Level" chart
    if (window.ChartManager) {
        console.log("Replacing ChartManager.extractModelLevelDetailsData with safe version");
        window.ChartManager.extractModelLevelDetailsData = window.SafeModelComparisonChartManager.extractModelLevelDetailsData;
    }
    
    // Monitor model comparison selector clicks
    const modelDetailButton = document.querySelector('#model_comparison_selector [data-chart-type="details"]');
    if (modelDetailButton) {
        modelDetailButton.addEventListener('click', function() {
            console.log("Details chart button clicked, applying fix");
            // Check if we need to reinitialize the chart
            const chartElement = document.getElementById('model-level-details-chart-plot');
            if (chartElement && chartElement.innerHTML.includes('No Data Available')) {
                // Reinitialize the chart with our fixed function
                if (typeof ChartManager !== 'undefined' && 
                    typeof ChartManager.initializeModelLevelDetailsChart === 'function') {
                    setTimeout(function() {
                        console.log("Reinitializing model level details chart");
                        ChartManager.initializeModelLevelDetailsChart('model-level-details-chart-plot');
                    }, 300);
                }
            }
        });
    }
});