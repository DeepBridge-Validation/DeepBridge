/**
 * Utilitários para extração de dados de relatórios de robustez
 * 
 * Este módulo concentra todas as funções para acessar e extrair dados
 * das várias estruturas possíveis nos resultados de testes de robustez.
 */

const RobustnessDataExtractor = {
    /**
     * Obtém os dados principais de robustez de qualquer localização possível no relatório
     */
    getRobustnessData: function(reportData) {
        if (!reportData) return null;
        
        // Tenta todos os caminhos possíveis para encontrar os resultados de robustez
        let results = null;
        
        // Caminho 1: Padrão
        if (reportData.results && 
            reportData.results.robustness && 
            reportData.results.robustness.results) {
            results = reportData.results.robustness.results;
        } 
        // Caminho 2: Direto
        else if (reportData.robustness && reportData.robustness.results) {
            results = reportData.robustness.results;
        }
        // Caminho 3: Modelo primário no nível superior
        else if (reportData.primary_model) {
            results = {
                primary_model: reportData.primary_model
            };
        }
        
        return results;
    },

    /**
     * Obtém dados do modelo primário
     */
    getPrimaryModelData: function(reportData) {
        const results = this.getRobustnessData(reportData);
        return results?.primary_model || null;
    },

    /**
     * Extrai dados de importância de características para gráficos e tabelas
     */
    getFeatureImportanceData: function(reportData) {
        // Primeiro verificar variáveis globais do template
        if (window.feature_importance && window.model_feature_importance) {
            return {
                featureImportance: window.feature_importance,
                modelFeatureImportance: window.model_feature_importance,
                featureSubset: window.feature_subset || []
            };
        }

        // Caso contrário, tentar obter dos resultados
        const primaryModel = this.getPrimaryModelData(reportData);
        if (!primaryModel) return null;

        return {
            featureImportance: primaryModel.feature_importance || {},
            modelFeatureImportance: primaryModel.model_feature_importance || {},
            featureSubset: primaryModel.feature_subset || []
        };
    },

    /**
     * Extrai dados de perturbação para gráficos e tabelas
     */
    getPerturbationData: function(reportData) {
        const primaryModel = this.getPrimaryModelData(reportData);
        if (!primaryModel || !primaryModel.raw || !primaryModel.raw.by_level) {
            return null;
        }

        // Extrai níveis e os ordena
        const levels = Object.keys(primaryModel.raw.by_level)
            .map(level => parseFloat(level))
            .sort((a, b) => a - b);

        // Processa dados para cada nível
        const byLevel = {};
        const scores = [];
        const worstScores = [];
        
        levels.forEach(level => {
            const levelStr = level.toString();
            const levelData = primaryModel.raw.by_level[levelStr];
            
            // Extrai scores e impactos
            const score = this._extractMeanScore(levelData);
            scores.push(score);
            
            const worstScore = this._extractWorstScore(levelData);
            worstScores.push(worstScore);
            
            byLevel[levelStr] = {
                score: score,
                worstScore: worstScore,
                impact: this._calculateImpact(primaryModel.base_score, score)
            };
        });

        // Obtém dados de modelos alternativos, se disponíveis
        const alternativeModels = {};
        const robustnessResults = this.getRobustnessData(reportData);
        
        if (robustnessResults && robustnessResults.alternative_models) {
            Object.entries(robustnessResults.alternative_models).forEach(([name, model]) => {
                if (model.raw && model.raw.by_level) {
                    alternativeModels[name] = this._processAlternativeModel(model, levels);
                }
            });
        }

        return {
            modelName: primaryModel.model_type || primaryModel.model_name || 'Primary Model',
            baseScore: primaryModel.base_score || 0,
            metric: primaryModel.metric || 'Score',
            levels: levels,
            scores: scores,
            worstScores: worstScores,
            byLevel: byLevel,
            alternativeModels: alternativeModels
        };
    },

    /**
     * Extrai worst_score de qualquer localização possível na estrutura de dados
     */
    _extractWorstScore: function(levelData) {
        if (!levelData) return null;
        
        // Tenta múltiplos caminhos específicos em ordem de probabilidade
        
        // Caminho 1: runs.all_features[0].worst_score (mais comum)
        if (levelData.runs?.all_features?.[0]?.worst_score !== undefined) {
            return levelData.runs.all_features[0].worst_score;
        }
        
        // Caminho 2: overall_result.all_features.worst_score
        if (levelData.overall_result?.all_features?.worst_score !== undefined) {
            return levelData.overall_result.all_features.worst_score;
        }
        
        // Caminho 3: runs.feature_subset[0].worst_score
        if (levelData.runs?.feature_subset?.[0]?.worst_score !== undefined) {
            return levelData.runs.feature_subset[0].worst_score;
        }
        
        // Caminho 4: overall_result.feature_subset.worst_score
        if (levelData.overall_result?.feature_subset?.worst_score !== undefined) {
            return levelData.overall_result.feature_subset.worst_score;
        }
        
        // Caminho 5: Verificar todos os arrays em runs
        if (levelData.runs) {
            for (const key in levelData.runs) {
                const runData = levelData.runs[key];
                if (Array.isArray(runData) && runData.length > 0 && runData[0].worst_score !== undefined) {
                    return runData[0].worst_score;
                }
            }
        }
        
        // Caminho 6: Verificar todos os objetos em overall_result
        if (levelData.overall_result) {
            for (const key in levelData.overall_result) {
                const resultData = levelData.overall_result[key];
                if (resultData && typeof resultData === 'object' && resultData.worst_score !== undefined) {
                    return resultData.worst_score;
                }
            }
        }
        
        // Nenhum worst_score encontrado
        return null;
    },

    /**
     * Extrai mean_score (score médio) de uma estrutura de nível
     */
    _extractMeanScore: function(levelData) {
        if (!levelData) return null;
        
        // Verificar overall_result primeiro
        if (levelData.overall_result) {
            // Verificar all_features
            if (levelData.overall_result.all_features && 
                levelData.overall_result.all_features.mean_score !== undefined) {
                return levelData.overall_result.all_features.mean_score;
            }
            
            // Verificar feature_subset
            if (levelData.overall_result.feature_subset && 
                levelData.overall_result.feature_subset.mean_score !== undefined) {
                return levelData.overall_result.feature_subset.mean_score;
            }
            
            // Verificar outros campos no overall_result
            for (const key in levelData.overall_result) {
                if (levelData.overall_result[key] && 
                    levelData.overall_result[key].mean_score !== undefined) {
                    return levelData.overall_result[key].mean_score;
                }
            }
        }
        
        // Verificar em runs.all_features se overall_result não tiver os dados
        if (levelData.runs?.all_features?.[0]?.perturbed_score !== undefined) {
            return levelData.runs.all_features[0].perturbed_score;
        }
        
        // Não encontrado
        return null;
    },

    /**
     * Calcula o impacto baseado no score base e no score perturbado
     */
    _calculateImpact: function(baseScore, score) {
        if (!baseScore || baseScore === 0) return 0;
        return Math.max(0, baseScore - score) / baseScore;
    },

    /**
     * Processa dados de um modelo alternativo 
     */
    _processAlternativeModel: function(model, levelOptions) {
        const scores = [];
        const worstScores = [];
        const byLevel = {};
        
        // Usar os mesmos níveis do modelo primário como referência
        levelOptions.forEach(level => {
            const levelStr = level.toString();
            const levelData = model.raw?.by_level?.[levelStr];
            
            // Se não tivermos dados para este nível, usar null
            if (!levelData) {
                scores.push(null);
                worstScores.push(null);
                return;
            }
            
            // Extrair scores
            const score = this._extractMeanScore(levelData);
            scores.push(score);
            
            const worstScore = this._extractWorstScore(levelData);
            worstScores.push(worstScore);
            
            byLevel[levelStr] = {
                score: score,
                worstScore: worstScore,
                impact: this._calculateImpact(model.base_score, score)
            };
        });
        
        return {
            modelName: model.model_type || model.model_name || 'Alternative Model',
            baseScore: model.base_score || 0,
            scores: scores,
            worstScores: worstScores,
            byLevel: byLevel
        };
    }
};