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
   * Generate synthetic distribution data based on mean and worst scores
   * @param {Object} perturbationData - Extracted perturbation data
   * @return {Object} Distribution data with synthetic samples
   */
  generateDistributionData(perturbationData) {
    if (!perturbationData || !perturbationData.levels || perturbationData.levels.length === 0) {
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
    
    // Generate distribution data for primary model
    perturbationData.levels.forEach((level, index) => {
      const meanScore = perturbationData.scores[index];
      const worstScore = perturbationData.worstScores[index];
      
      // Calculate synthetic bounds
      // Q1 = mean - (mean - worst) / 2, min = worst, Q3 = mean + (mean - worst) / 4, max = mean + (mean - worst) / 2
      const variability = meanScore - worstScore;
      
      distribution.primaryModel.distributions.push({
        level: level,
        mean: meanScore,
        median: meanScore * 0.99, // Slightly lower than mean
        min: worstScore,
        max: Math.min(1, meanScore + (variability / 2)),
        q1: meanScore - (variability / 2),
        q3: meanScore + (variability / 4),
        outliers: []
      });
    });
    
    // Generate distribution data for alternative models
    if (perturbationData.alternativeModels) {
      Object.entries(perturbationData.alternativeModels).forEach(([name, model]) => {
        const alternativeModel = {
          name: name,
          distributions: []
        };
        
        perturbationData.levels.forEach((level, index) => {
          if (model.scores[index] === null) {
            // No data for this level
            alternativeModel.distributions.push(null);
            return;
          }
          
          const meanScore = model.scores[index];
          const worstScore = model.worstScores[index] || meanScore * 0.95; // Fallback if no worst score
          
          // Calculate synthetic bounds
          const variability = meanScore - worstScore;
          
          alternativeModel.distributions.push({
            level: level,
            mean: meanScore,
            median: meanScore * 0.99, // Slightly lower than mean
            min: worstScore,
            max: Math.min(1, meanScore + (variability / 2)),
            q1: meanScore - (variability / 2),
            q3: meanScore + (variability / 4),
            outliers: []
          });
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