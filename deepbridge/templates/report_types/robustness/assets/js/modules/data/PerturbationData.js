/**
 * Perturbation Data Handler
 * 
 * Specialized handler for extracting and processing perturbation data
 * from the robustness report structure.
 */

class PerturbationData {
  /**
   * Initialize the perturbation data handler
   */
  constructor() {
    // Default values for fallbacks
    this.defaultLevels = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
  }
  
  /**
   * Extract perturbation data from report data
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted perturbation data or null if not available
   */
  extract(reportData) {
    if (!reportData) return null;
    
    let perturbationData = null;
    
    // Check if data is directly in reportData
    if (reportData.perturbation_results && reportData.base_score) {
      perturbationData = this.extractPerturbationData(reportData);
    } 
    // Check if data is in a primary_model property
    else if (reportData.primary_model && reportData.primary_model.perturbation_results) {
      perturbationData = this.extractPerturbationData(reportData.primary_model);
      
      // Add alternative models if available
      if (reportData.alternative_models) {
        perturbationData.alternativeModels = {};
        
        Object.entries(reportData.alternative_models).forEach(([name, modelData]) => {
          if (modelData && modelData.perturbation_results) {
            perturbationData.alternativeModels[name] = this.extractPerturbationDataForAlternativeModel(
              modelData, 
              perturbationData.levels
            );
          }
        });
      }
    }
    
    return perturbationData;
  }
  
  /**
   * Extract perturbation data for the primary model
   * @param {Object} modelData - Model data containing perturbation results
   * @return {Object} Extracted and normalized perturbation data
   */
  extractPerturbationData(modelData) {
    const results = modelData.perturbation_results || {};
    const baseScore = modelData.base_score || 0;
    const metric = modelData.metric || 'Score';
    const modelName = modelData.model_name || 'Primary Model';
    
    // Get levels from perturbation results or use defaults
    const levels = Object.keys(results)
      .filter(key => !isNaN(parseFloat(key)))
      .map(key => parseFloat(key))
      .sort((a, b) => a - b);
    
    if (levels.length === 0) {
      return null;
    }
    
    // Extract scores and worst scores
    const scores = [];
    const worstScores = [];
    const byLevel = {};
    
    levels.forEach(level => {
      const levelStr = level.toString();
      const levelData = results[levelStr] || {};
      
      // Get mean score for this level
      const score = levelData.mean_score !== undefined ? levelData.mean_score : 
                    levelData.score !== undefined ? levelData.score : 
                    null;
      
      // Get worst score for this level
      const worstScore = levelData.worst_score !== undefined ? levelData.worst_score : 
                         levelData.min_score !== undefined ? levelData.min_score : 
                         score * 0.9; // Fallback: estimate worst score as 90% of mean score
      
      // Calculate impact (percentage decrease from base score)
      const impact = score !== null ? (baseScore - score) / baseScore : null;
      
      scores.push(score);
      worstScores.push(worstScore);
      
      // Store all level data for detailed views
      byLevel[levelStr] = {
        score: score,
        worstScore: worstScore,
        impact: impact,
        // Add any other level-specific data available
        stdDev: levelData.std_dev || levelData.standard_deviation || null,
        samples: levelData.samples || null,
        quantiles: levelData.quantiles || null
      };
    });
    
    // Calculate robustness score as average of normalized scores
    const robustnessScore = scores.reduce((sum, score) => sum + (score / baseScore), 0) / scores.length;
    
    return {
      levels: levels,
      scores: scores,
      worstScores: worstScores,
      baseScore: baseScore,
      metric: metric,
      modelName: modelName,
      robustnessScore: robustnessScore,
      byLevel: byLevel
    };
  }
  
  /**
   * Extract perturbation data for an alternative model
   * @param {Object} modelData - Alternative model data
   * @param {Array} primaryLevels - Perturbation levels from the primary model
   * @return {Object} Extracted and normalized perturbation data
   */
  extractPerturbationDataForAlternativeModel(modelData, primaryLevels) {
    const results = modelData.perturbation_results || {};
    const baseScore = modelData.base_score || 0;
    
    // Extract scores and worst scores, aligned with primary model levels
    const scores = [];
    const worstScores = [];
    
    primaryLevels.forEach(level => {
      const levelStr = level.toString();
      const levelData = results[levelStr] || {};
      
      // Get mean score for this level (or null if not available)
      const score = levelData.mean_score !== undefined ? levelData.mean_score : 
                    levelData.score !== undefined ? levelData.score : 
                    null;
      
      // Get worst score for this level (or null if not available)
      const worstScore = levelData.worst_score !== undefined ? levelData.worst_score : 
                         levelData.min_score !== undefined ? levelData.min_score : 
                         score !== null ? score * 0.9 : null; // Fallback: estimate worst score as 90% of mean score
      
      scores.push(score);
      worstScores.push(worstScore);
    });
    
    // Calculate robustness score as average of normalized scores
    const validScores = scores.filter(score => score !== null);
    const robustnessScore = validScores.length > 0 ? 
      validScores.reduce((sum, score) => sum + (score / baseScore), 0) / validScores.length : 
      null;
    
    return {
      baseScore: baseScore,
      scores: scores,
      worstScores: worstScores,
      robustnessScore: robustnessScore
    };
  }
  
  /**
   * Check if distribution data is present in the perturbation data
   * @param {Object} perturbationData - Extracted perturbation data
   * @return {Object|null} Distribution data or null if not available
   */
  generateDistributionData(perturbationData) {
    if (!perturbationData || !perturbationData.levels || perturbationData.levels.length === 0) {
      return null;
    }
    
    // Check if we have actual distribution data in the perturbation data
    // If detailed distribution data is not available, return null
    if (!perturbationData.byLevel || 
        !Object.values(perturbationData.byLevel).some(level => level.quantiles || level.samples)) {
      console.warn("No detailed distribution data available for boxplot visualization");
      return null;
    }
    
    const distribution = {
      levels: perturbationData.levels,
      primaryModel: {
        name: perturbationData.modelName,
        distributions: []
      },
      alternativeModels: []
    };
    
    // Extract real distribution data for primary model if available
    let hasDistributionData = false;
    
    perturbationData.levels.forEach((level, index) => {
      const levelStr = level.toString();
      const levelData = perturbationData.byLevel[levelStr];
      
      if (levelData && (levelData.quantiles || levelData.samples)) {
        hasDistributionData = true;
        
        // Use actual distribution data
        distribution.primaryModel.distributions.push({
          level: level,
          mean: levelData.score || perturbationData.scores[index],
          median: levelData.quantiles ? levelData.quantiles.median : null,
          min: levelData.quantiles ? levelData.quantiles.min : null,
          max: levelData.quantiles ? levelData.quantiles.max : null,
          q1: levelData.quantiles ? levelData.quantiles.q1 : null,
          q3: levelData.quantiles ? levelData.quantiles.q3 : null,
          outliers: levelData.quantiles ? (levelData.quantiles.outliers || []) : []
        });
      } else {
        // If no distribution data, push null
        distribution.primaryModel.distributions.push(null);
      }
    });
    
    // If no distribution data found, return null
    if (!hasDistributionData) {
      console.warn("No detailed distribution data available for visualization");
      return null;
    }
    
    // Extract distribution data for alternative models if available
    if (perturbationData.alternativeModels) {
      Object.entries(perturbationData.alternativeModels).forEach(([name, model]) => {
        if (!model.byLevel || !Object.values(model.byLevel).some(level => level.quantiles || level.samples)) {
          return; // Skip models without distribution data
        }
        
        const alternativeModel = {
          name: name,
          distributions: []
        };
        
        perturbationData.levels.forEach((level, index) => {
          const levelStr = level.toString();
          const levelData = model.byLevel ? model.byLevel[levelStr] : null;
          
          if (levelData && (levelData.quantiles || levelData.samples)) {
            alternativeModel.distributions.push({
              level: level,
              mean: levelData.score || model.scores[index],
              median: levelData.quantiles ? levelData.quantiles.median : null,
              min: levelData.quantiles ? levelData.quantiles.min : null,
              max: levelData.quantiles ? levelData.quantiles.max : null,
              q1: levelData.quantiles ? levelData.quantiles.q1 : null,
              q3: levelData.quantiles ? levelData.quantiles.q3 : null,
              outliers: levelData.quantiles ? (levelData.quantiles.outliers || []) : []
            });
          } else {
            alternativeModel.distributions.push(null);
          }
        });
        
        distribution.alternativeModels.push(alternativeModel);
      });
    }
    
    return distribution;
  }
  
  /**
   * Calculate performance metrics for the perturbation results
   * @param {Object} perturbationData - Extracted perturbation data
   * @return {Object} Performance metrics
   */
  calculatePerformanceMetrics(perturbationData) {
    if (!perturbationData || !perturbationData.scores || perturbationData.scores.length === 0) {
      return {
        avgScore: null,
        minScore: null,
        maxScore: null,
        avgImpact: null,
        maxImpact: null,
        robustnessScore: null
      };
    }
    
    const baseScore = perturbationData.baseScore || 1.0;
    const scores = perturbationData.scores;
    const worstScores = perturbationData.worstScores;
    
    // Calculate metrics
    const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const minScore = Math.min(...worstScores);
    const maxScore = Math.max(...scores);
    
    // Calculate impacts
    const impacts = scores.map(score => (baseScore - score) / baseScore);
    const avgImpact = impacts.reduce((sum, impact) => sum + impact, 0) / impacts.length;
    const maxImpact = Math.max(...impacts);
    
    // Calculate robustness score if not already available
    const robustnessScore = perturbationData.robustnessScore || 
                           (1 - avgImpact); // Fallback: 1 - average impact
    
    return {
      avgScore,
      minScore,
      maxScore,
      avgImpact,
      maxImpact,
      robustnessScore
    };
  }
}

export default PerturbationData;