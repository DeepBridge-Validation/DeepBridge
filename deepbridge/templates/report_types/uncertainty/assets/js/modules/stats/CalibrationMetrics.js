// CalibrationMetrics.js - placeholder
/**
 * Calibration Metrics
 * 
 * Statistical metrics for assessing the calibration of uncertainty estimates,
 * including Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
 * and reliability diagrams.
 */

class CalibrationMetrics {
    /**
     * Initialize the calibration metrics calculator
     */
    constructor() {
      // Default number of bins for binning predictions
      this.defaultBinCount = 10;
    }
    
    /**
     * Calculate Expected Calibration Error (ECE)
     * @param {Array} predictedProbabilities - Predicted probabilities or confidence scores
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @param {number} binCount - Number of bins for calculating ECE
     * @return {Object} ECE and binned data
     */
    calculateECE(predictedProbabilities, actualOutcomes, binCount = this.defaultBinCount) {
      if (!predictedProbabilities || !actualOutcomes || 
          predictedProbabilities.length !== actualOutcomes.length ||
          predictedProbabilities.length === 0) {
        return {
          ece: null,
          bins: []
        };
      }
      
      // Create bins
      const bins = this.createBins(predictedProbabilities, actualOutcomes, binCount);
      
      // Calculate ECE
      let ece = 0;
      let totalSamples = predictedProbabilities.length;
      
      bins.forEach(bin => {
        if (bin.sampleCount > 0) {
          const binWeight = bin.sampleCount / totalSamples;
          const binError = Math.abs(bin.predictedProbability - bin.observedFrequency);
          ece += binWeight * binError;
        }
      });
      
      return {
        ece,
        bins
      };
    }
    
    /**
     * Calculate Maximum Calibration Error (MCE)
     * @param {Array} predictedProbabilities - Predicted probabilities or confidence scores
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @param {number} binCount - Number of bins for calculating MCE
     * @return {Object} MCE and binned data
     */
    calculateMCE(predictedProbabilities, actualOutcomes, binCount = this.defaultBinCount) {
      if (!predictedProbabilities || !actualOutcomes || 
          predictedProbabilities.length !== actualOutcomes.length ||
          predictedProbabilities.length === 0) {
        return {
          mce: null,
          bins: []
        };
      }
      
      // Create bins
      const bins = this.createBins(predictedProbabilities, actualOutcomes, binCount);
      
      // Calculate MCE
      let mce = 0;
      
      bins.forEach(bin => {
        if (bin.sampleCount > 0) {
          const binError = Math.abs(bin.predictedProbability - bin.observedFrequency);
          mce = Math.max(mce, binError);
        }
      });
      
      return {
        mce,
        bins
      };
    }
    
    /**
     * Create bins for calibration assessment
     * @param {Array} predictedProbabilities - Predicted probabilities or confidence scores
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @param {number} binCount - Number of bins to create
     * @return {Array} Array of bin objects
     */
    createBins(predictedProbabilities, actualOutcomes, binCount) {
      // Initialize bins
      const bins = [];
      const binWidth = 1.0 / binCount;
      
      for (let i = 0; i < binCount; i++) {
        bins.push({
          lowerBound: i * binWidth,
          upperBound: (i + 1) * binWidth,
          predictedProbability: 0,
          observedFrequency: 0,
          sampleCount: 0,
          samples: []
        });
      }
      
      // Assign samples to bins
      for (let i = 0; i < predictedProbabilities.length; i++) {
        const prob = predictedProbabilities[i];
        const outcome = actualOutcomes[i];
        
        // Find the appropriate bin
        let binIndex = Math.floor(prob / binWidth);
        
        // Handle edge case for probability of 1.0
        if (binIndex === binCount) {
          binIndex = binCount - 1;
        }
        
        // Skip if out of range
        if (binIndex < 0 || binIndex >= binCount) continue;
        
        // Update bin statistics
        bins[binIndex].sampleCount += 1;
        bins[binIndex].samples.push({
          probability: prob,
          outcome: outcome
        });
        
        // Update sum for later averaging
        bins[binIndex].predictedProbability += prob;
        bins[binIndex].observedFrequency += outcome;
      }
      
      // Calculate average values for each bin
      bins.forEach(bin => {
        if (bin.sampleCount > 0) {
          bin.predictedProbability /= bin.sampleCount;
          bin.observedFrequency /= bin.sampleCount;
        }
      });
      
      return bins;
    }
    
    /**
     * Calculate Brier Score
     * @param {Array} predictedProbabilities - Predicted probabilities
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @return {number} Brier score
     */
    calculateBrierScore(predictedProbabilities, actualOutcomes) {
      if (!predictedProbabilities || !actualOutcomes || 
          predictedProbabilities.length !== actualOutcomes.length ||
          predictedProbabilities.length === 0) {
        return null;
      }
      
      let sumSquaredError = 0;
      
      for (let i = 0; i < predictedProbabilities.length; i++) {
        const error = predictedProbabilities[i] - actualOutcomes[i];
        sumSquaredError += error * error;
      }
      
      return sumSquaredError / predictedProbabilities.length;
    }
    
    /**
     * Calculate Log Loss (Cross-entropy loss)
     * @param {Array} predictedProbabilities - Predicted probabilities
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @return {number} Log loss
     */
    calculateLogLoss(predictedProbabilities, actualOutcomes) {
      if (!predictedProbabilities || !actualOutcomes || 
          predictedProbabilities.length !== actualOutcomes.length ||
          predictedProbabilities.length === 0) {
        return null;
      }
      
      let sumLogLoss = 0;
      const epsilon = 1e-15; // Small constant to avoid log(0)
      
      for (let i = 0; i < predictedProbabilities.length; i++) {
        // Clip probabilities to avoid numerical issues
        const prob = Math.max(epsilon, Math.min(1 - epsilon, predictedProbabilities[i]));
        const outcome = actualOutcomes[i];
        
        // Calculate log loss: -[y*log(p) + (1-y)*log(1-p)]
        sumLogLoss += -(outcome * Math.log(prob) + (1 - outcome) * Math.log(1 - prob));
      }
      
      return sumLogLoss / predictedProbabilities.length;
    }
    
    /**
     * Generate reliability diagram data for visualization
     * @param {Array} predictedProbabilities - Predicted probabilities
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @param {number} binCount - Number of bins for the diagram
     * @return {Object} Reliability diagram data
     */
    generateReliabilityDiagram(predictedProbabilities, actualOutcomes, binCount = this.defaultBinCount) {
      // Create bins
      const bins = this.createBins(predictedProbabilities, actualOutcomes, binCount);
      
      // Extract data for visualization
      const binMidpoints = [];
      const observedFrequencies = [];
      const predictedProbabilitiesAvg = [];
      const sampleCounts = [];
      
      bins.forEach(bin => {
        if (bin.sampleCount > 0) {
          const midpoint = (bin.lowerBound + bin.upperBound) / 2;
          binMidpoints.push(midpoint);
          observedFrequencies.push(bin.observedFrequency);
          predictedProbabilitiesAvg.push(bin.predictedProbability);
          sampleCounts.push(bin.sampleCount);
        }
      });
      
      return {
        binMidpoints,
        observedFrequencies,
        predictedProbabilitiesAvg,
        sampleCounts,
        bins
      };
    }
    
    /**
     * Apply temperature scaling to calibrate probabilities
     * @param {Array} predictedProbabilities - Uncalibrated probabilities
     * @param {number} temperature - Temperature parameter (T > 0)
     * @return {Array} Calibrated probabilities
     */
    applyTemperatureScaling(predictedProbabilities, temperature) {
      if (!predictedProbabilities || temperature <= 0) {
        return [];
      }
      
      // Apply temperature scaling: p' = p^(1/T)
      return predictedProbabilities.map(prob => {
        // Handle edge cases
        if (prob <= 0) return 0;
        if (prob >= 1) return 1;
        
        // Apply temperature scaling
        return Math.pow(prob, 1/temperature);
      });
    }
    
    /**
     * Find optimal temperature parameter for calibration
     * @param {Array} predictedProbabilities - Uncalibrated probabilities
     * @param {Array} actualOutcomes - Actual binary outcomes (0/1)
     * @param {number} maxIter - Maximum iterations for optimization
     * @return {Object} Optimal temperature and calibrated probabilities
     */
    findOptimalTemperature(predictedProbabilities, actualOutcomes, maxIter = 100) {
      if (!predictedProbabilities || !actualOutcomes || 
          predictedProbabilities.length !== actualOutcomes.length ||
          predictedProbabilities.length === 0) {
        return {
          temperature: 1.0,
          calibratedProbabilities: [...predictedProbabilities]
        };
      }
      
      // Simple grid search for optimal temperature
      let bestTemp = 1.0;
      let bestLoss = this.calculateLogLoss(predictedProbabilities, actualOutcomes);
      
      // Search temperatures from 0.1 to 10 in 0.1 increments
      for (let t = 0.1; t <= 10; t += 0.1) {
        const calibratedProbs = this.applyTemperatureScaling(predictedProbabilities, t);
        const loss = this.calculateLogLoss(calibratedProbs, actualOutcomes);
        
        if (loss < bestLoss) {
          bestLoss = loss;
          bestTemp = t;
        }
      }
      
      // Apply the best temperature
      const calibratedProbabilities = this.applyTemperatureScaling(predictedProbabilities, bestTemp);
      
      return {
        temperature: bestTemp,
        calibratedProbabilities
      };
    }
  }
  
  export default CalibrationMetrics;