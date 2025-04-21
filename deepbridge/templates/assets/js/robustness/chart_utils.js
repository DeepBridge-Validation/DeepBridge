/**
 * Utilitários para criação e configuração de gráficos nos relatórios de robustez
 * 
 * Este módulo concentra funções relacionadas ao estilo e configuração
 * de gráficos, garantindo um aspecto visual consistente.
 */

const ChartUtils = {
    // Paletas de cores
    colors: {
        primary: '#1b78de',         // Azul principal
        secondary: '#2ecc71',       // Verde secundário
        accent: '#ff7300',          // Laranja destaque
        warning: '#e74c3c',         // Vermelho alerta
        alternativeModels: [
            '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', 
            '#a65628', '#f781bf', '#999999', '#ffd92f'
        ]
    },
    
    /**
     * Cria um layout padrão com estilos consistentes
     */
    createStandardLayout: function(title, xAxisTitle, yAxisTitle, options = {}) {
        return {
            title: title,
            xaxis: {
                title: xAxisTitle,
                ...options.xaxis
            },
            yaxis: {
                title: yAxisTitle,
                ...options.yaxis
            },
            margin: {
                l: options.leftMargin || 60,
                r: options.rightMargin || 30,
                t: options.topMargin || 50,
                b: options.bottomMargin || 60
            },
            legend: {
                orientation: options.legendOrientation || 'h',
                y: options.legendY || -0.2,
                x: options.legendX || 0.5,
                xanchor: options.legendXAnchor || 'center'
            },
            hovermode: options.hovermode || 'closest',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(248,249,250,0.5)',
            ...options.additionalProps
        };
    },
    
    /**
     * Cria traces padrão para gráficos de perturbação
     */
    createPerturbationTraces: function(data) {
        const traces = [];
        
        // Trace do modelo primário
        traces.push({
            x: data.levels,
            y: data.scores,
            mode: 'lines+markers',
            type: 'scatter',
            name: data.modelName,
            marker: {
                size: 8,
                color: this.colors.primary
            },
            line: {
                width: 3,
                color: this.colors.primary
            }
        });
        
        // Adiciona modelos alternativos, se disponíveis
        if (data.alternativeModels) {
            let colorIndex = 0;
            
            for (const [modelName, modelData] of Object.entries(data.alternativeModels)) {
                if (!modelData.scores || modelData.scores.length === 0) continue;
                
                // Filtra out valores null
                const levels = [];
                const scores = [];
                
                data.levels.forEach((level, i) => {
                    if (modelData.scores[i] !== null) {
                        levels.push(level);
                        scores.push(modelData.scores[i]);
                    }
                });
                
                if (scores.length === 0) continue;
                
                traces.push({
                    x: levels,
                    y: scores,
                    mode: 'lines+markers',
                    type: 'scatter',
                    name: modelName,
                    marker: {
                        size: 7,
                        symbol: 'circle',
                        color: this.colors.alternativeModels[colorIndex % this.colors.alternativeModels.length]
                    },
                    line: {
                        width: 2,
                        dash: 'solid',
                        color: this.colors.alternativeModels[colorIndex % this.colors.alternativeModels.length]
                    }
                });
                
                colorIndex++;
            }
        }
        
        return traces;
    },
    
    /**
     * Cria traces para gráfico de worst scores
     */
    createWorstScoreTraces: function(data) {
        const traces = [];
        
        // Trace do modelo primário
        if (data.worstScores && data.worstScores.length > 0) {
            // Filtra valores null
            const levels = [];
            const scores = [];
            
            data.levels.forEach((level, i) => {
                if (data.worstScores[i] !== null) {
                    levels.push(level);
                    scores.push(data.worstScores[i]);
                }
            });
            
            if (scores.length > 0) {
                traces.push({
                    x: levels,
                    y: scores,
                    mode: 'lines+markers',
                    type: 'scatter',
                    name: data.modelName,
                    marker: {
                        size: 8,
                        color: this.colors.primary
                    },
                    line: {
                        width: 3,
                        color: this.colors.primary
                    }
                });
            }
        }
        
        // Adiciona modelos alternativos, se disponíveis
        if (data.alternativeModels) {
            let colorIndex = 0;
            
            for (const [modelName, modelData] of Object.entries(data.alternativeModels)) {
                if (!modelData.worstScores || modelData.worstScores.length === 0) continue;
                
                // Filtra valores null
                const levels = [];
                const scores = [];
                
                data.levels.forEach((level, i) => {
                    if (modelData.worstScores[i] !== null) {
                        levels.push(level);
                        scores.push(modelData.worstScores[i]);
                    }
                });
                
                if (scores.length === 0) continue;
                
                traces.push({
                    x: levels,
                    y: scores,
                    mode: 'lines+markers',
                    type: 'scatter',
                    name: modelName,
                    marker: {
                        size: 7,
                        symbol: 'circle',
                        color: this.colors.alternativeModels[colorIndex % this.colors.alternativeModels.length]
                    },
                    line: {
                        width: 2,
                        dash: 'solid',
                        color: this.colors.alternativeModels[colorIndex % this.colors.alternativeModels.length]
                    }
                });
                
                colorIndex++;
            }
        }
        
        return traces;
    },
    
    /**
     * Cria traces para gráfico de importância de características
     */
    createFeatureImportanceTraces: function(features, importanceValues, options = {}) {
        // Combina dados para ordenar
        const combined = features.map((feature, i) => ({
            feature,
            importance: importanceValues[i] || 0,
            inSubset: options.inSubset ? options.inSubset.includes(feature) : false
        }));
        
        // Ordena por importância
        combined.sort((a, b) => b.importance - a.importance);
        
        // Limita ao número especificado ou 15 por padrão
        const limit = options.limit || 15;
        const topFeatures = combined.slice(0, limit);
        
        // Extrai dados ordenados
        const sortedFeatures = topFeatures.map(item => item.feature);
        const sortedImportance = topFeatures.map(item => item.importance);
        
        // Define cores com base no subconjunto ótimo
        const colors = topFeatures.map(item => 
            item.inSubset ? 
            'rgba(46, 204, 113, 0.8)' : 
            'rgba(46, 204, 113, 0.4)'
        );
        
        return [{
            x: sortedImportance,
            y: sortedFeatures,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: options.colors || colors,
                line: {
                    color: options.lineColor || 'rgba(46, 204, 113, 1.0)',
                    width: 1
                }
            }
        }];
    },
    
    /**
     * Cria traces para gráfico de barras de impactos
     */
    createImpactBarTraces: function(data) {
        // Extrai dados de impacto por nível
        const levels = data.levels;
        const impacts = levels.map(level => {
            const levelData = data.byLevel[level.toString()];
            return levelData ? levelData.impact : 0;
        });
        
        // Cria trace para o gráfico de barras
        return [{
            x: levels,
            y: impacts,
            type: 'bar',
            name: data.modelName,
            marker: {
                color: this.colors.primary,
                line: {
                    color: 'rgba(27, 120, 222, 1.0)',
                    width: 1
                }
            }
        }];
    },
    
    /**
     * Converte hex para RGB para uso no Plotly
     */
    hexToRgb: function(hex) {
        // Remove o hash, se existir
        hex = hex.replace('#', '');
        
        // Parse dos valores hex
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        
        return {r, g, b};
    }
};