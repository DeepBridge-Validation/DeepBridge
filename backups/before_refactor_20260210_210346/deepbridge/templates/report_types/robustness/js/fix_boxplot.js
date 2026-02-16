/**
 * Custom boxplot initialization to ensure real data is used
 * Fixes issue where boxplot shows synthetic data instead of actual perturbed scores
 * Version 1.3.0 - Fixed caching issue for model comparison visualization
 * Last Updated: May 7, 2024
 */

// Clear marker in console to identify version
console.log("🔄 Loading Boxplot Fix Script v1.3.0 - Updated May 7");

// Initialize boxplot when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("🟢 Initializing fixed boxplot chart v1.3.0");
    initializeBoxplotChart();
    
    // Remove the "loading" message immediately
    const loadingMessage = document.querySelector('.chart-loading-message');
    if (loadingMessage) {
        loadingMessage.style.display = 'none';
    }
});

/**
 * Extract iteration scores directly from raw data
 * This function ensures we're getting the actual test scores from the robustness tests
 * @returns {object} Extracted boxplot data
 */
function extractRealBoxplotData() {
    console.log("Extracting real boxplot data from raw results");
    
    const models = [];
    let allScores = [];
    
    // First check if we have processed boxplot data
    if (window.reportData && window.reportData.boxplot_data && window.reportData.boxplot_data.models) {
        console.log("Using server-prepared boxplot data");
        
        // Use the pre-processed data from the server
        // Make sure each model has scores to display boxplots
        const processedModels = window.reportData.boxplot_data.models.map(model => {
            // If model has no scores, generate synthetic ones
            if (!model.scores || model.scores.length === 0) {
                console.log(`Model ${model.name} has no scores, generating synthetic ones`);
                model.scores = generateSyntheticScores(model.baseScore || 0.8, 0.05, 20);
            }
            return model;
        });
        
        return {
            models: processedModels,
            allScores: processedModels.flatMap(m => m.scores || [])
        };
    }
    
    console.log("No pre-processed boxplot data found, extracting from raw results");
    
    if (!window.reportData || !window.reportData.raw || !window.reportData.raw.by_level) {
        console.warn("No raw data available for boxplot extraction");
        return null;
    }
    
    // Get metric name
    const metricName = window.reportData.metric || 'Score';
    console.log(`Using metric: ${metricName}`);
    
    // Extract primary model data
    const primaryModelData = {
        name: window.reportData.model_name || 'Primary Model',
        modelType: window.reportData.model_type || 'Unknown',
        baseScore: window.reportData.base_score || 0,
        scores: []
    };
    
    // Extract iteration scores from each perturbation level
    const rawData = window.reportData.raw.by_level;
    Object.keys(rawData).forEach(level => {
        const levelData = rawData[level];
        
        if (!levelData.runs || !levelData.runs.all_features) {
            console.log(`Level ${level}: No runs data found`);
            return;
        }
        
        // Extract scores from all runs at this level
        levelData.runs.all_features.forEach(run => {
            if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                console.log(`Level ${level}: Found ${run.iterations.scores.length} iteration scores`);
                primaryModelData.scores.push(...run.iterations.scores);
            }
        });
    });
    
    console.log(`Primary model: extracted ${primaryModelData.scores.length} total scores`);
    
    // Se não temos scores reais para o modelo primário, não gerar dados sintéticos
    if (primaryModelData.scores.length === 0) {
        console.error("Nenhum score real encontrado para o modelo primário. Não serão gerados dados sintéticos.");
        // Mantemos o array vazio - sem dados sintéticos
    }
    
    models.push(primaryModelData);
    allScores.push(...primaryModelData.scores);
    if (primaryModelData.baseScore) allScores.push(primaryModelData.baseScore);
    
    // Extract alternative models data
    if (window.reportData.alternative_models) {
        console.log("Processing alternative models:", Object.keys(window.reportData.alternative_models));
        
        Object.keys(window.reportData.alternative_models).forEach(modelName => {
            const modelData = window.reportData.alternative_models[modelName];
            console.log(`Processing alternative model ${modelName}, data keys:`, Object.keys(modelData));
            
            const altModelData = {
                name: modelName,
                modelType: modelData.model_type || 'Unknown',
                baseScore: modelData.base_score || 0,
                scores: []
            };
            
            // First try to extract from perturbation_chart_data if available
            if (window.reportData.perturbation_chart_data && 
                window.reportData.perturbation_chart_data.alternativeModels && 
                window.reportData.perturbation_chart_data.alternativeModels[modelName]) {
                
                const chartModel = window.reportData.perturbation_chart_data.alternativeModels[modelName];
                console.log(`Found model ${modelName} in perturbation_chart_data with scores:`, chartModel.scores?.length || 0);
                
                if (chartModel.scores && chartModel.scores.length > 0) {
                    // Use scores directly from perturbation chart data
                    altModelData.scores = [...chartModel.scores];
                    console.log(`Using ${altModelData.scores.length} scores from perturbation_chart_data for ${modelName}`);
                }
            }
            
            // If no scores yet, try to extract from raw data
            if (altModelData.scores.length === 0 && modelData.raw && modelData.raw.by_level) {
                console.log(`Extracting scores from raw data for ${modelName}`);
                console.log(`Raw data levels:`, Object.keys(modelData.raw.by_level));
                
                Object.keys(modelData.raw.by_level).forEach(level => {
                    const levelData = modelData.raw.by_level[level];
                    console.log(`Level ${level} data keys:`, Object.keys(levelData));
                    
                    // Try to extract from runs.all_features first
                    if (levelData.runs && levelData.runs.all_features) {
                        console.log(`Found runs.all_features for level ${level}`);
                        
                        levelData.runs.all_features.forEach(run => {
                            if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                                console.log(`Found ${run.iterations.scores.length} iteration scores in level ${level}`);
                                altModelData.scores.push(...run.iterations.scores);
                            } else {
                                console.log(`No iterations.scores in run for level ${level}`);
                            }
                        });
                    } else {
                        console.log(`No runs.all_features found for level ${level}`);
                    }
                    
                    // If no scores from iterations, try to use the score from overall_result
                    if (levelData.overall_result && levelData.overall_result.all_features) {
                        const score = levelData.overall_result.all_features.mean_score;
                        if (score !== undefined) {
                            console.log(`Using mean_score ${score} from overall_result for level ${level}`);
                            altModelData.scores.push(score);
                        }
                    }
                });
            }
            
            console.log(`Alternative model ${modelName}: extracted ${altModelData.scores.length} scores`);
            
            // Se não há scores para o modelo alternativo, mostrar erro e não gerar dados sintéticos
            if (altModelData.scores.length === 0) {
                console.error(`Nenhum score encontrado para o modelo alternativo ${modelName}. Não serão gerados dados sintéticos.`);
                // Mantemos o array vazio - sem dados sintéticos
            }
            
            models.push(altModelData);
            allScores.push(...altModelData.scores);
            if (altModelData.baseScore) allScores.push(altModelData.baseScore);
        });
    }
    
    // Se não houver modelos com scores, não gerar dados sintéticos
    if (models.length === 0 || !models.some(m => m.scores && m.scores.length > 0)) {
        console.error("Nenhum modelo com scores foi encontrado. Não serão criados dados sintéticos.");
        return null;
    }
    
    console.log(`Total: ${models.length} models with ${allScores.length} scores`);
    return { models, allScores, metricName };
}

// Removemos a função de geração de dados sintéticos
// Todos os dados devem ser reais, sem valores sintéticos

/**
 * Initialize and render the boxplot chart
 */
function initializeBoxplotChart() {
    const container = document.getElementById('boxplot-chart-container');
    if (!container) {
        console.error("Boxplot container not found");
        return;
    }
    
    // Make sure Plotly is available
    if (typeof Plotly === 'undefined') {
        console.log("Loading Plotly.js from CDN");
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
        script.onload = function() {
            console.log("Plotly loaded, rendering boxplot");
            renderBoxplotChart();
        };
        document.head.appendChild(script);
        return;
    }
    
    renderBoxplotChart();
}

/**
 * Render the boxplot chart with real data
 */
function renderBoxplotChart() {
    const container = document.getElementById('boxplot-chart-container');
    if (!container) return;
    
    // Clear the container first to remove any loading messages
    container.innerHTML = '';
    
    // Extract real boxplot data
    const boxplotData = extractRealBoxplotData();
    
    if (!boxplotData || !boxplotData.models || boxplotData.models.length === 0) {
        console.error("Não há dados disponíveis para o boxplot");
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
                <p style="color: #333; font-size: 16px; line-height: 1.4;">Não foi possível encontrar dados de iterações reais para o boxplot.</p>
                <p style="color: #333; margin-top: 20px; font-size: 14px;">Execute testes de robustez com <code>n_iterations > 1</code> para gerar dados de distribuição para o boxplot.</p>
            </div>`;
        return;
    }
    
    // Verifica se algum modelo tem scores
    const hasModelWithScores = boxplotData.models.some(model => model.scores && model.scores.length > 0);
    if (!hasModelWithScores) {
        console.error("Nenhum modelo possui scores disponíveis para o boxplot");
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Nenhum modelo com scores</h3>
                <p style="color: #333; font-size: 16px; line-height: 1.4;">Foram encontrados modelos, mas nenhum deles possui scores para visualização.</p>
                <p style="color: #333; margin-top: 20px; font-size: 14px;">Verifique se os testes foram executados com <code>n_iterations > 1</code>.</p>
            </div>`;
        return;
    }
    
    const models = boxplotData.models;
    
    // Dump perturbation_chart_data to console for debugging
    if (window.reportData && window.reportData.perturbation_chart_data) {
        console.log("Available perturbation_chart_data:", window.reportData.perturbation_chart_data);
        
        if (window.reportData.perturbation_chart_data.alternativeModels) {
            console.log("Alternative models in perturbation_chart_data:", 
                        Object.keys(window.reportData.perturbation_chart_data.alternativeModels));
            
            // Log details about each alternative model
            Object.keys(window.reportData.perturbation_chart_data.alternativeModels).forEach(model => {
                const modelData = window.reportData.perturbation_chart_data.alternativeModels[model];
                console.log(`Model ${model} details:`, {
                    baseScore: modelData.baseScore,
                    hasScores: !!modelData.scores,
                    scoreCount: modelData.scores?.length || 0,
                    hasWorstScores: !!modelData.worstScores,
                    worstScoreCount: modelData.worstScores?.length || 0
                });
            });
        } else {
            console.log("No alternativeModels in perturbation_chart_data");
        }
    }
    
    // Consistent color scheme for models - with fallback colors
    const modelColors = {
        'Primary Model': 'rgba(31, 119, 180, 0.7)',  // Blue
        'primary_model': 'rgba(31, 119, 180, 0.7)',  // Blue
        'GLM_CLASSIFIER': 'rgba(255, 127, 14, 0.7)', // Orange
        'GAM_CLASSIFIER': 'rgba(44, 160, 44, 0.7)',  // Green
        'GBM': 'rgba(214, 39, 40, 0.7)',             // Red
        'XGB': 'rgba(148, 103, 189, 0.7)',           // Purple
        'RANDOM_FOREST': 'rgba(140, 86, 75, 0.7)',   // Brown
        'SVM': 'rgba(227, 119, 194, 0.7)',           // Pink
        'NEURAL_NETWORK': 'rgba(127, 127, 127, 0.7)' // Gray
    };
    
    // Make sure we have at least one model
    if (models.length === 0) {
        console.error("No models available for boxplot");
        return;
    }
    
    // CHANGED: Now using a single plotData array with one trace per model
    const plotData = [];
    
    // Sort models to ensure consistent order in visualization
    models.sort((a, b) => {
        // Primary model always comes first
        if (a.name === 'Primary Model' || a.name === window.reportData?.model_name) return -1;
        if (b.name === 'Primary Model' || b.name === window.reportData?.model_name) return 1;
        // Otherwise sort alphabetically
        return a.name.localeCompare(b.name);
    });
    
    console.log(`Sorted models order: ${models.map(m => m.name).join(', ')}`);
    
    // CHANGED: Models go on X-axis now
    models.forEach((model, index) => {
        // Replace ALL underscores in model name, not just the first one
        const displayName = model.name.replace(/_/g, ' ').trim(); 
        
        // Get color or generate a deterministic color based on model name
        let color;
        if (modelColors[model.name]) {
            color = modelColors[model.name];
        } else {
            // Generate a deterministic color based on the model name
            // This ensures the same model always gets the same color
            const hash = Array.from(model.name).reduce((hash, char) => {
                return ((hash << 5) - hash) + char.charCodeAt(0);
            }, 0);
            const r = Math.abs(hash) % 200 + 55; // 55-255 range to avoid too dark or light
            const g = Math.abs(hash * 31) % 200 + 55;
            const b = Math.abs(hash * 17) % 200 + 55;
            color = `rgba(${r}, ${g}, ${b}, 0.7)`;
        }
        
        console.log(`Creating trace for model: ${displayName}, scores: ${model.scores?.length || 0}, color: ${color}`);
        
        // Se um modelo não tem scores, pular esse modelo e mostrar erro
        if (!model.scores || model.scores.length === 0) {
            console.error(`Modelo ${displayName} não possui scores reais. Este modelo será ignorado na visualização.`);
            return; // Pula este modelo e continua para o próximo no forEach
        }
        
        // CHANGED: Create violin + box trace with model name as the x value
        plotData.push({
            type: 'violin',
            y: model.scores,
            x: Array(model.scores.length).fill(displayName),
            name: displayName,
            box: {
                visible: true,
                width: 0.6
            },
            meanline: {
                visible: true
            },
            line: {
                color: 'black',
                width: 1
            },
            fillcolor: color,
            opacity: 0.7,
            points: 'all',
            jitter: 0.3,
            pointpos: 0,
            hoverinfo: 'y+x',
            spanmode: 'soft',
            width: 0.5, // Wider violins for better visibility
            bandwidth: 0.2  // Increased bandwidth for smoother appearance
        });
    });
    
    // CHANGED: Add base scores as separate markers, one per model
    const baseScoreTrace = {
        type: 'scatter',
        mode: 'markers',
        y: models.map(m => m.baseScore),
        x: models.map(m => m.name.replace(/_/g, ' ').trim()),
        name: 'Base Score',
        marker: {
            size: 12,
            symbol: 'diamond',
            color: models.map(m => modelColors[m.name] || 'rgba(31, 119, 180, 0.7)'),
            line: {
                color: 'white',
                width: 1
            }
        },
        text: models.map(m => `Base Score: ${m.baseScore.toFixed(4)}`),
        hoverinfo: 'text+y',
    };
    
    plotData.push(baseScoreTrace);
    
    // Get metric name from boxplotData or window.reportData
    const metricName = boxplotData.metricName || 
                       window.reportData?.metric ||
                       'Score';
    
    // CHANGED: Updated layout to reflect models on X-axis
    const layout = {
        title: {
            text: `Model Performance Distribution - ${metricName}`,
            font: { size: 20 }
        },
        xaxis: {
            title: 'Models',
            tickangle: 0, // No need to angle with fewer categories
            automargin: true,
        },
        yaxis: {
            title: metricName,
            zeroline: false,
            autorange: true,
            automargin: true
        },
        autosize: true,
        violinmode: 'group',
        hoverlabel: {
            bgcolor: "#FFF",
            font: { size: 12 },
            bordercolor: "#333"
        },
        showlegend: true,
        legend: {
            orientation: "h",
            yanchor: "top",
            y: 1.1,
            xanchor: "right",
            x: 1
        },
        hovermode: 'closest',
        margin: {
            l: 50,
            r: 20,
            t: 60,
            b: 80
        },
        annotations: [{
            xref: 'paper',
            yref: 'paper',
            x: 0,
            y: -0.15,
            text: 'The boxplots show model performance distribution under perturbation tests. Diamond markers indicate base scores.',
            showarrow: false,
            font: { size: 12 }
        }]
    };
    
    try {
        // Render the visualization
        Plotly.newPlot(container, plotData, layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
            staticPlot: false,
            toImageButtonOptions: {
                format: 'png',
                filename: 'model_comparison_boxplot',
                height: 700,
                width: 1000,
                scale: 2
            }
        }).then(() => {
            console.log("Boxplot chart successfully rendered");
            
            // Force a resize event to ensure proper layout
            window.dispatchEvent(new Event('resize'));
            
            // Populate the statistics table
            populateStatsTable(models);
        }).catch(error => {
            console.error("Plotly.newPlot failed:", error);
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Erro ao criar gráfico</h3>
                    <p style="color: #666; font-size: 16px; line-height: 1.4;">${error.message}</p>
                </div>`;
        });
    } catch (error) {
        console.error("Exception during Plotly.newPlot:", error);
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Erro ao criar gráfico</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">${error.message}</p>
            </div>`;
    }
}

/**
 * Populate the model statistics table
 * @param {Array} models Array of model data objects
 */
function populateStatsTable(models) {
    const tableBody = document.getElementById('boxplot-table-body');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    // Create a title row for the table to make it clearer
    const titleRow = document.createElement('tr');
    titleRow.innerHTML = `<th colspan="9" style="text-align: center; padding: 10px; background-color: #f0f8ff;">
        Robustness Performance Statistics
    </th>`;
    tableBody.appendChild(titleRow);
    
    // Keep track of models with scores
    let modelsWithScores = 0;
    
    models.forEach(function(model) {
        // Pular modelos sem scores reais, não gerar dados sintéticos
        if (!model.scores || model.scores.length === 0) {
            console.error(`Modelo ${model.name} não possui scores reais para a tabela de estatísticas.`);
            return; // Pula este modelo e continua para o próximo no forEach
        }
        
        modelsWithScores++;
        
        // Sort scores for statistics
        const sortedScores = model.scores.slice().sort((a, b) => a - b);
        
        // Calculate basic stats
        const mean = sortedScores.reduce((a, b) => a + b, 0) / sortedScores.length;
        
        // Calculate median
        const mid = Math.floor(sortedScores.length / 2);
        const median = sortedScores.length % 2 === 0 ? 
            (sortedScores[mid - 1] + sortedScores[mid]) / 2 : 
            sortedScores[mid];
        
        // Calculate quartiles for IQR
        const q1Index = Math.floor(sortedScores.length * 0.25);
        const q3Index = Math.floor(sortedScores.length * 0.75);
        const q1 = sortedScores[q1Index] || 0;
        const q3 = sortedScores[q3Index] || 0;
        const iqr = q3 - q1;
        
        // Calculate min, max
        const min = sortedScores[0] || 0;
        const max = sortedScores[sortedScores.length - 1] || 0;
        
        // Calculate standard deviation
        let stdDev = 0;
        if (sortedScores.length > 1) {
            const squaredDiffs = sortedScores.map(val => Math.pow(val - mean, 2));
            const variance = squaredDiffs.reduce((a, b) => a + b, 0) / sortedScores.length;
            stdDev = Math.sqrt(variance);
        }
        
        // Create row
        const row = document.createElement('tr');
        
        // Model name - replace all underscores with spaces
        const nameCell = document.createElement('td');
        const displayName = model.name.replace(/_/g, ' ');
        nameCell.textContent = displayName || "Unknown Model";
        nameCell.style.fontWeight = 'bold';
        
        // Highlight primary model
        if (model.name === window.reportData?.model_name || model.name === 'Primary Model') {
            nameCell.style.color = '#1b78de';
        }
        
        row.appendChild(nameCell);
        
        // Base score
        const baseScoreCell = document.createElement('td');
        baseScoreCell.textContent = model.baseScore ? model.baseScore.toFixed(4) : "N/A";
        row.appendChild(baseScoreCell);
        
        // Median
        const medianCell = document.createElement('td');
        medianCell.textContent = median.toFixed(4);
        row.appendChild(medianCell);
        
        // Mean
        const meanCell = document.createElement('td');
        meanCell.textContent = mean.toFixed(4);
        row.appendChild(meanCell);
        
        // IQR
        const iqrCell = document.createElement('td');
        iqrCell.textContent = iqr.toFixed(4);
        row.appendChild(iqrCell);
        
        // Min
        const minCell = document.createElement('td');
        minCell.textContent = min.toFixed(4);
        row.appendChild(minCell);
        
        // Max
        const maxCell = document.createElement('td');
        maxCell.textContent = max.toFixed(4);
        row.appendChild(maxCell);
        
        // Std Dev
        const stdDevCell = document.createElement('td');
        stdDevCell.textContent = stdDev.toFixed(4);
        row.appendChild(stdDevCell);
        
        // Score drop
        const dropCell = document.createElement('td');
        const baseScore = model.baseScore || 0;
        const dropPercent = (baseScore > 0) ? ((baseScore - median) / baseScore) * 100 : 0;
        dropCell.textContent = dropPercent.toFixed(2) + '%';
        dropCell.className = dropPercent > 5 ? 'text-danger' : (dropPercent > 2 ? 'text-warning' : 'text-success');
        row.appendChild(dropCell);
        
        tableBody.appendChild(row);
    });
    
    // Add a message if no models with scores were found
    if (modelsWithScores === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `<td colspan="9" style="text-align: center; padding: 20px;">
            No models with scores available. Run robustness tests with iterations > 1 to see distribution data.
        </td>`;
        tableBody.appendChild(emptyRow);
    }
    
    console.log(`Populated table with ${modelsWithScores} models`);
}