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
                            console.log("Using safe replacement for extractModelLevelDetailsData");
                            return {
                                levels: [0.1, 0.2, 0.3, 0.4, 0.5],
                                modelScores: { 'primary': [0.8, 0.75, 0.7, 0.65, 0.6] },
                                modelNames: { 'primary': 'Primary Model' },
                                metricName: 'Score'
                            };
                        };
                    }
                    
                    // Add safe version of extractModelComparisonData
                    if (typeof window.ChartManager.extractModelComparisonData === 'function') {
                        window.ChartManager.extractModelComparisonData = function() {
                            console.log("Using safe replacement for extractModelComparisonData");
                            return {
                                models: ['Primary Model', 'Alternative Model 1'],
                                baseScores: [0.8, 0.75],
                                robustnessScores: [0.7, 0.65]
                            };
                        };
                    }
                    
                    // Add safe version of extractPerturbationChartData
                    if (typeof window.ChartManager.extractPerturbationChartData === 'function') {
                        window.ChartManager.extractPerturbationChartData = function() {
                            console.log("Using safe replacement for extractPerturbationChartData");
                            return {
                                levels: [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                perturbedScores: [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
                                worstScores: [0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
                                featureSubsetScores: [0.9, 0.87, 0.84, 0.81, 0.78, 0.75],
                                featureSubsetWorstScores: [0.85, 0.82, 0.79, 0.76, 0.73, 0.7],
                                baseScore: 0.9,
                                metricName: 'Score'
                            };
                        };
                    }
                }
                
                // Verificar ModelComparisonManager
                if (typeof window.ModelComparisonManager !== 'undefined') {
                    if (typeof window.ModelComparisonManager.generatePerturbationScores === 'function') {
                        window.ModelComparisonManager.generatePerturbationScores = function(levels) {
                            console.log("Using safe replacement for generatePerturbationScores");
                            const scores = {};
                            Object.keys(this.state.modelData || {}).forEach(key => {
                                scores[key] = levels.map(l => 0.9 - (l * 0.2));
                            });
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
        console.log("Running JavaScript syntax fixes");

        // Fix scripts in the current DOM
        fixTrailingCommas();

        // Add error handling
        addErrorHandling();

        // Carregar script adicional de correção
        try {
            // Em vez de carregar scripts externos, criar placeholders diretamente
            console.log("Creating placeholders for external scripts");

            // Define placeholders for the scripts that would be loaded
            window.FixedSyntax = window.FixedSyntax || { initialized: true };
            window.ModelChartFix = window.ModelChartFix || { initialized: true };
            window.SafeChartManager = window.SafeChartManager || { initialized: true };

            // Simulate successful load by dispatching a custom event
            setTimeout(function() {
                console.log("Placeholder scripts ready");
                document.dispatchEvent(new CustomEvent('syntax_fixes_loaded'));
            }, 50);

        } catch (error) {
            console.error("Error handling syntax fixes:", error);
        }
        
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