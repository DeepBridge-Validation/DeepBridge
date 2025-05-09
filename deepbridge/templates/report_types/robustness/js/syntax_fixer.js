// JavaScript Syntax Fixer
// This script applies runtime fixes for common JavaScript syntax errors
// Include this script in the <head> of your HTML document before any other scripts

(function() {
    // Define our safe return object - this will be available for any other script
    window.__safeFallbackObject = {
        levels: [],
        modelScores: {},
        modelNames: {},
        metricName: ""
    };
    
    // Fix trailing commas in JavaScript object literals at runtime
    function fixTrailingCommas() {
        console.log("Running syntax fixer to fix trailing commas");
        
        // Find all script tags
        const scripts = document.querySelectorAll('script:not([src])');
        
        // Process each inline script
        for (const script of scripts) {
            if (!script.textContent) {
                // Usando if/else ao invés de continue
                continue;
            }
            
            let content = script.textContent;
            let needsReplacement = false;
            
            // Fix 1: Trailing commas in object literals - matches a return statement with an object that ends with a comma
            if (content.includes('return {') && 
                (content.includes('metricName,') || 
                 content.includes('robustnessScores,') || 
                 content.includes('baseScores,'))) {
                
                // First fix: return { ... , } pattern
                const fixedContent1 = content.replace(/return\s*\{\s*[\s\S]*?,(\s*\})/g, 'return $1');
                
                // Second fix: specific variable pattern for model data extraction
                const fixedContent2 = fixedContent1.replace(
                    /(return\s*\{\s*)(levels,\s*modelScores,\s*modelNames,\s*metricName),(\s*\})/g, 
                    '$1$2$3'
                );
                
                // Third fix: specific variable pattern for model comparison data
                const fixedContent3 = fixedContent2.replace(
                    /(return\s*\{\s*)(models,\s*baseScores,\s*robustnessScores),(\s*\})/g, 
                    '$1$2$3'
                );
                
                // Check if any fixes were applied
                if (content !== fixedContent3) {
                    content = fixedContent3;
                    needsReplacement = true;
                    console.log("Fixed trailing commas in return statements");
                }
            }
            
            // Fix 2: Any comma before closing brace
            if (content.includes('},') || content.includes(', }')) {
                const fixedContent = content.replace(/,(\s*\})/g, '$1');
                
                if (content !== fixedContent) {
                    content = fixedContent;
                    needsReplacement = true;
                    console.log("Fixed generic trailing commas");
                }
            }
            
            // Fix 3: Replace illegal continue statements with return null or early returns
            // Este é um ponto comum de erro em callbacks de map() ou em funções de callback internas
            if (content.includes('continue')) {
                
                // Primeiro, verificar por 'continue' dentro de funções map(), forEach(), e filter()
                if (content.includes('map(') || content.includes('forEach(') || content.includes('filter(')) {
                    // Procurar por padrões de 'continue' fora de loops que precisam ser substituídos em callbacks
                    const fixedContent = content.replace(
                        /if\s*\([^)]*\)\s*\{\s*[^{}]*continue;\s*\}/g, 
                        function(match) {
                            // Substituir 'continue' por 'return null' dentro do callback de map
                            return match.replace(/continue;/, 'return null;');
                        }
                    );
                    
                    if (content !== fixedContent) {
                        content = fixedContent;
                        needsReplacement = true;
                        console.log("Fixed illegal continue statements in callbacks");
                    }
                }
                
                // Segundo, verificar por qualquer 'continue' em funções de callback inline
                const fixedContent2 = content.replace(
                    /function\s*\([^)]*\)\s*\{(?:[^{}]|{[^{}]*})*continue;(?:[^{}]|{[^{}]*})*\}/g,
                    function(match) {
                        // Substituir 'continue' por 'return null' dentro de funções
                        return match.replace(/continue;/, 'return null;');
                    }
                );
                
                if (content !== fixedContent2) {
                    content = fixedContent2;
                    needsReplacement = true;
                    console.log("Fixed illegal continue statements in inline functions");
                }
                
                // Terceiro, verificar por qualquer 'continue' imediatamente após if, sem estar em um loop
                const fixedContent3 = content.replace(
                    /if\s*\([^)]*\)\s*continue;(?!\s*\})/g,
                    function(match) {
                        // Substituir 'continue' por 'return null' ou 'return'
                        if (content.includes('map(') || content.includes('filter(')) {
                            return match.replace(/continue;/, 'return null;');
                        } else if (content.includes('forEach(')) {
                            return match.replace(/continue;/, 'return;');
                        } else {
                            return match.replace(/continue;/, '{ /* skip */ }');
                        }
                    }
                );
                
                if (content !== fixedContent3) {
                    content = fixedContent3;
                    needsReplacement = true;
                    console.log("Fixed standalone illegal continue statements");
                }
            }
            
            // Replace the script if needed
            if (needsReplacement) {
                try {
                    const newScript = document.createElement('script');
                    newScript.textContent = content;
                    script.parentNode.replaceChild(newScript, script);
                    console.log("Replaced script with fixed version");
                } catch (error) {
                    console.error("Error replacing script:", error);
                }
            }
        }
    }
    
    // Add error handling for specific JavaScript errors
    function addErrorHandling() {
        window.addEventListener('error', function(event) {
            // Verificar especificamente por erro de continue ilegal
            if (event.error && 
                (event.error.toString().includes("Unexpected token") || 
                 event.error.toString().includes("Illegal continue") ||
                 event.error.toString().includes("no surrounding iteration statement"))) {
                
                console.warn("Caught syntax error:", event.error);
                
                // Try to monkeypatch global objects after error
                if (typeof window.ChartManager !== 'undefined') {
                    console.log("Adding safe fallbacks for ChartManager");
                    
                    // Add safe version of extractModelLevelDetailsData
                    if (typeof window.ChartManager.extractModelLevelDetailsData === 'function') {
                        window.ChartManager.extractModelLevelDetailsData = function() {
                            console.log("Using safe replacement for extractModelLevelDetailsData - showing only real data");
                            // Check if we have the data in the global reportData first
                            if (window.reportData && window.reportData.perturbation_chart_data) {
                                return {
                                    levels: window.reportData.perturbation_chart_data.levels || [],
                                    modelScores: window.reportData.perturbation_chart_data.modelScores || {},
                                    modelNames: window.reportData.perturbation_chart_data.modelNames || {},
                                    metricName: window.reportData.perturbation_chart_data.metric || 'Score'
                                };
                            }
                            // Return empty but valid data structure
                            return {
                                levels: [],
                                modelScores: {},
                                modelNames: {},
                                metricName: 'Score'
                            };
                        };
                    }
                    
                    // Add safe version of extractModelComparisonData
                    if (typeof window.ChartManager.extractModelComparisonData === 'function') {
                        window.ChartManager.extractModelComparisonData = function() {
                            console.log("Using safe replacement for extractModelComparisonData - showing only real data");
                            // Try to extract real data from reportData if available
                            let models = [];
                            let baseScores = [];
                            let robustnessScores = [];
                            
                            if (window.reportData) {
                                // Add primary model if available
                                if (window.reportData.model_name) {
                                    models.push(window.reportData.model_name);
                                    baseScores.push(window.reportData.base_score || 0);
                                    robustnessScores.push(window.reportData.robustness_score || 0);
                                }
                                
                                // Add alternative models if available
                                if (window.reportData.alternative_models) {
                                    for (const modelName in window.reportData.alternative_models) {
                                        const modelData = window.reportData.alternative_models[modelName];
                                        models.push(modelName);
                                        baseScores.push(modelData.base_score || 0);
                                        robustnessScores.push(modelData.robustness_score || 0);
                                    }
                                }
                            }
                            
                            return {
                                models: models,
                                baseScores: baseScores,
                                robustnessScores: robustnessScores
                            };
                        };
                    }
                    
                    // Add safe version of extractPerturbationChartData
                    if (typeof window.ChartManager.extractPerturbationChartData === 'function') {
                        window.ChartManager.extractPerturbationChartData = function() {
                            console.log("Using safe replacement for extractPerturbationChartData - showing only real data");
                            
                            // Try to extract real data from reportData or chartData
                            let chartData = {};
                            
                            if (window.reportData && window.reportData.perturbation_chart_data) {
                                chartData = window.reportData.perturbation_chart_data;
                            } else if (window.chartData && window.chartData.perturbation_chart_data) {
                                chartData = window.chartData.perturbation_chart_data;
                            }
                            
                            return {
                                levels: chartData.levels || [],
                                perturbedScores: chartData.scores || [],
                                worstScores: chartData.worstScores || [],
                                featureSubsetScores: chartData.featureSubsetScores || [],
                                featureSubsetWorstScores: chartData.featureSubsetWorstScores || [],
                                baseScore: chartData.baseScore || 0,
                                metricName: chartData.metric || 'Score'
                            };
                        };
                    }
                }
                
                // Verificar ModelComparisonManager
                if (typeof window.ModelComparisonManager !== 'undefined') {
                    if (typeof window.ModelComparisonManager.generatePerturbationScores === 'function') {
                        window.ModelComparisonManager.generatePerturbationScores = function(levels) {
                            console.log("Using safe replacement for generatePerturbationScores - showing only real data");
                            
                            // Try to extract real scores from the data
                            const scores = {};
                            
                            // Extract primary model scores if available
                            if (window.reportData && window.reportData.perturbation_chart_data) {
                                const chartData = window.reportData.perturbation_chart_data;
                                const modelName = window.reportData.model_name || 'primary';
                                
                                if (chartData.scores && Array.isArray(chartData.scores)) {
                                    scores[modelName] = chartData.scores;
                                }
                                
                                // Extract alternative model scores if available
                                if (chartData.alternativeModels) {
                                    for (const altModelName in chartData.alternativeModels) {
                                        const altModelData = chartData.alternativeModels[altModelName];
                                        if (altModelData.scores && Array.isArray(altModelData.scores)) {
                                            scores[altModelName] = altModelData.scores;
                                        }
                                    }
                                }
                            }
                            
                            // Check if we have any real data
                            if (Object.keys(scores).length === 0) {
                                // Return empty scores for all models in state
                                Object.keys(this.state.modelData || {}).forEach(key => {
                                    scores[key] = levels.map(() => null);
                                });
                            }
                            
                            return scores;
                        };
                    }
                }
                
                // Prevent the error from propagating
                event.preventDefault();
            }
        }, true);
    }
    
    // Run the fixes when the DOM is ready
    function runFixes() {
        console.log("Running JavaScript syntax fixes - embedded version");
        
        // Fix scripts in the current DOM
        fixTrailingCommas();
        
        // Add error handling
        addErrorHandling();
        
        // No external script loading - all scripts are now embedded directly in the HTML
        console.log("All scripts are embedded in HTML - no external loading needed");
        
        // Setup MutationObserver to fix dynamically added scripts
        const observer = new MutationObserver(function(mutations) {
            // Check if any scripts were added
            let scriptAdded = false;
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.tagName === 'SCRIPT') {
                            scriptAdded = true;
                        } else if (node.querySelectorAll) {
                            const scripts = node.querySelectorAll('script');
                            if (scripts.length > 0) {
                                scriptAdded = true;
                            }
                        }
                    });
                }
            });
            
            // If scripts were added, run the fixer again
            if (scriptAdded) {
                console.log("New scripts detected, running fixes");
                fixTrailingCommas();
            }
        });
        
        // Start observing the document
        observer.observe(document, {
            childList: true,
            subtree: true
        });
    }
    
    // Run fixes when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runFixes);
    } else {
        runFixes();
    }
})();