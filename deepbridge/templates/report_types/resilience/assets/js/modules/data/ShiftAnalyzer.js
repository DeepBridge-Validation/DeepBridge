// ShiftAnalyzer.js - placeholder
/**
 * Shift Analyzer
 * 
 * Utilities for analyzing distribution shifts and calculating
 * distribution distance metrics.
 */

class ShiftAnalyzer {
    /**
     * Initialize the shift analyzer
     */
    constructor() {
      // Configuration for analyzer
      this.config = {
        binCount: 20,  // Number of bins for histogram-based calculations
        minSamples: 10 // Minimum number of samples needed for calculations
      };
    }
    
    /**
     * Calculate distance metrics between two distributions
     * @param {Array} original - Original distribution values
     * @param {Array} shifted - Shifted distribution values
     * @return {Object} Distance metrics
     */
    calculateDistanceMetrics(original, shifted) {
      if (!original || !shifted || 
          original.length < this.config.minSamples || 
          shifted.length < this.config.minSamples) {
        return {
          kl_divergence: null,
          js_divergence: null,
          wasserstein: null,
          hellinger: null
        };
      }
      
      // Calculate different distance metrics
      const metrics = {
        kl_divergence: this.calculateKLDivergence(original, shifted),
        js_divergence: this.calculateJSDivergence(original, shifted),
        wasserstein: this.calculateWassersteinDistance(original, shifted),
        hellinger: this.calculateHellingerDistance(original, shifted)
      };
      
      return metrics;
    }
    
    /**
     * Calculate Kullback-Leibler divergence
     * @param {Array} original - Original distribution values
     * @param {Array} shifted - Shifted distribution values
     * @return {number} KL divergence
     */
    calculateKLDivergence(original, shifted) {
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...original), Math.min(...shifted));
      const max = Math.max(Math.max(...original), Math.max(...shifted));
      const binCount = this.config.binCount;
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      original.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      shifted.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histQ[binIndex]++;
      });
      
      // Normalize histograms to get probability distributions
      const sumP = histP.reduce((a, b) => a + b, 0);
      const sumQ = histQ.reduce((a, b) => a + b, 0);
      
      const probP = histP.map(count => count / sumP);
      const probQ = histQ.map(count => count / sumQ);
      
      // Calculate KL divergence: sum(p(i) * log(p(i) / q(i)))
      let divergence = 0;
      const epsilon = 1e-10; // Small value to avoid division by zero
      
      for (let i = 0; i < binCount; i++) {
        if (probP[i] > 0) {
          divergence += probP[i] * Math.log((probP[i] + epsilon) / (probQ[i] + epsilon));
        }
      }
      
      return divergence;
    }
    
    /**
     * Calculate Jensen-Shannon divergence
     * @param {Array} original - Original distribution values
     * @param {Array} shifted - Shifted distribution values
     * @return {number} JS divergence
     */
    calculateJSDivergence(original, shifted) {
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...original), Math.min(...shifted));
      const max = Math.max(Math.max(...original), Math.max(...shifted));
      const binCount = this.config.binCount;
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      original.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      shifted.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histQ[binIndex]++;
      });
      
      // Normalize histograms to get probability distributions
      const sumP = histP.reduce((a, b) => a + b, 0);
      const sumQ = histQ.reduce((a, b) => a + b, 0);
      
      const probP = histP.map(count => count / sumP);
      const probQ = histQ.map(count => count / sumQ);
      
      // Calculate midpoint distribution M = (P + Q) / 2
      const probM = probP.map((val, i) => (val + probQ[i]) / 2);
      
      // Calculate KL(P||M) and KL(Q||M)
      let klPM = 0;
      let klQM = 0;
      const epsilon = 1e-10;
      
      for (let i = 0; i < binCount; i++) {
        if (probP[i] > 0) {
          klPM += probP[i] * Math.log((probP[i] + epsilon) / (probM[i] + epsilon));
        }
        
        if (probQ[i] > 0) {
          klQM += probQ[i] * Math.log((probQ[i] + epsilon) / (probM[i] + epsilon));
        }
      }
      
      // JS = 0.5 * (KL(P||M) + KL(Q||M))
      return 0.5 * (klPM + klQM);
    }
    
    /**
     * Calculate Wasserstein distance (Earth Mover's Distance)
     * @param {Array} original - Original distribution values
     * @param {Array} shifted - Shifted distribution values
     * @return {number} Wasserstein distance
     */
    calculateWassersteinDistance(original, shifted) {
      // Sort distributions
      const sortedP = [...original].sort((a, b) => a - b);
      const sortedQ = [...shifted].sort((a, b) => a - b);
      
      // Calculate empirical CDFs
      const n = sortedP.length;
      const m = sortedQ.length;
      
      // We need equal length samples, so resample to the minimum length
      const minLength = Math.min(n, m);
      const resampledP = [];
      const resampledQ = [];
      
      for (let i = 0; i < minLength; i++) {
        resampledP.push(sortedP[Math.floor(i * n / minLength)]);
        resampledQ.push(sortedQ[Math.floor(i * m / minLength)]);
      }
      
      // Calculate 1st Wasserstein distance (L1 norm)
      let distance = 0;
      for (let i = 0; i < minLength; i++) {
        distance += Math.abs(resampledP[i] - resampledQ[i]);
      }
      
      return distance / minLength;
    }
    
    /**
     * Calculate Hellinger distance
     * @param {Array} original - Original distribution values
     * @param {Array} shifted - Shifted distribution values
     * @return {number} Hellinger distance
     */
    calculateHellingerDistance(original, shifted) {
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...original), Math.min(...shifted));
      const max = Math.max(Math.max(...original), Math.max(...shifted));
      const binCount = this.config.binCount;
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      original.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      shifted.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histQ[binIndex]++;
      });
      
      // Normalize histograms to get probability distributions
      const sumP = histP.reduce((a, b) => a + b, 0);
      const sumQ = histQ.reduce((a, b) => a + b, 0);
      
      const probP = histP.map(count => count / sumP);
      const probQ = histQ.map(count => count / sumQ);
      
      // Calculate Hellinger distance: 1/sqrt(2) * sqrt(sum((sqrt(p) - sqrt(q))^2))
      let sumSquaredDiff = 0;
      
      for (let i = 0; i < binCount; i++) {
        const sqrtP = Math.sqrt(probP[i]);
        const sqrtQ = Math.sqrt(probQ[i]);
        sumSquaredDiff += Math.pow(sqrtP - sqrtQ, 2);
      }
      
      return Math.sqrt(sumSquaredDiff / 2);
    }
    
    /**
     * Classify shift type based on distribution properties
     * @param {Object} original - Original distribution data
     * @param {Object} shifted - Shifted distribution data
     * @return {string} Classified shift type
     */
    classifyShiftType(original, shifted) {
      if (!original || !shifted || 
          !original.values || !shifted.values ||
          original.values.length < this.config.minSamples || 
          shifted.values.length < this.config.minSamples) {
        return 'unknown';
      }
      
      // Extract statistical properties
      const originalMean = original.mean || this.calculateMean(original.values);
      const shiftedMean = shifted.mean || this.calculateMean(shifted.values);
      const originalStd = original.stdDev || this.calculateStdDev(original.values);
      const shiftedStd = shifted.stdDev || this.calculateStdDev(shifted.values);
      
      // Calculate relative changes
      const meanChange = Math.abs(shiftedMean - originalMean) / Math.abs(originalMean);
      const stdChange = Math.abs(shiftedStd - originalStd) / Math.abs(originalStd);
      
      // Classification thresholds
      const MEAN_THRESHOLD = 0.1;
      const STD_THRESHOLD = 0.1;
      
      // Classify shift type
      if (meanChange > MEAN_THRESHOLD && stdChange <= STD_THRESHOLD) {
        return 'covariate_shift'; // Mean shift, similar variance
      } else if (meanChange <= MEAN_THRESHOLD && stdChange > STD_THRESHOLD) {
        return 'concept_drift'; // Similar mean, different variance/shape
      } else if (meanChange > MEAN_THRESHOLD && stdChange > STD_THRESHOLD) {
        return 'dataset_shift'; // Both mean and variance/shape change
      } else {
        return 'minor_shift'; // Small changes in both
      }
    }
    
    /**
     * Calculate impact of shift on model performance
     * @param {Array} performanceOriginal - Performance metrics on original distribution
     * @param {Array} performanceShifted - Performance metrics on shifted distribution
     * @return {Object} Impact metrics
     */
    calculatePerformanceImpact(performanceOriginal, performanceShifted) {
      if (!performanceOriginal || !performanceShifted || 
          performanceOriginal.length === 0 || performanceShifted.length === 0) {
        return {
          absoluteChange: null,
          relativeChange: null,
          percentageChange: null,
          impactScore: null
        };
      }
      
      // Calculate average performance
      const avgOriginal = this.calculateMean(performanceOriginal);
      const avgShifted = this.calculateMean(performanceShifted);
      
      // Calculate impact metrics
      const absoluteChange = avgShifted - avgOriginal;
      const relativeChange = absoluteChange / avgOriginal;
      const percentageChange = relativeChange * 100;
      
      // Calculate impact score (0-1 scale, higher means more impact)
      // Normalized to [0,1] range with sigmoid function
      const impactScore = 2 / (1 + Math.exp(-Math.abs(relativeChange) * 5)) - 1;
      
      return {
        absoluteChange,
        relativeChange,
        percentageChange,
        impactScore
      };
    }
    
    /**
     * Calculate feature-wise shifts
     * @param {Object} originalData - Original dataset with features
     * @param {Object} shiftedData - Shifted dataset with features
     * @return {Object} Feature-wise shift metrics
     */
    calculateFeatureShifts(originalData, shiftedData) {
      if (!originalData || !shiftedData || !originalData.features || !shiftedData.features) {
        return {};
      }
      
      const featureShifts = {};
      const features = Object.keys(originalData.features);
      
      features.forEach(feature => {
        const originalValues = originalData.features[feature];
        const shiftedValues = shiftedData.features[feature];
        
        if (originalValues && shiftedValues && 
            originalValues.length >= this.config.minSamples && 
            shiftedValues.length >= this.config.minSamples) {
          
          // Calculate distance metrics for this feature
          const distanceMetrics = this.calculateDistanceMetrics(originalValues, shiftedValues);
          
          // Calculate statistical properties
          const originalMean = this.calculateMean(originalValues);
          const shiftedMean = this.calculateMean(shiftedValues);
          const originalStd = this.calculateStdDev(originalValues);
          const shiftedStd = this.calculateStdDev(shiftedValues);
          
          // Calculate relative changes
          const meanChange = (shiftedMean - originalMean) / Math.abs(originalMean);
          const stdChange = (shiftedStd - originalStd) / Math.abs(originalStd);
          
          // Store feature shift information
          featureShifts[feature] = {
            distanceMetrics,
            originalMean,
            shiftedMean,
            originalStd,
            shiftedStd,
            meanChange,
            stdChange,
            shiftMagnitude: Math.sqrt(Math.pow(meanChange, 2) + Math.pow(stdChange, 2))
          };
        }
      });
      
      return featureShifts;
    }
    
    /**
     * Calculate resilience score based on performance impact and shift magnitude
     * @param {Object} performanceImpact - Performance impact metrics
     * @param {Object} distanceMetrics - Distribution distance metrics
     * @return {number} Resilience score (0-1, higher is better)
     */
    calculateResilienceScore(performanceImpact, distanceMetrics) {
      if (!performanceImpact || !distanceMetrics || 
          performanceImpact.impactScore === null || 
          (distanceMetrics.js_divergence === null && 
           distanceMetrics.wasserstein === null)) {
        return null;
      }
      
      // Normalize distance metrics to [0,1] scale
      let normalizedDistance;
      
      // Prefer JS divergence, fallback to Wasserstein
      if (distanceMetrics.js_divergence !== null) {
        // JS is already in [0,1]
        normalizedDistance = distanceMetrics.js_divergence;
      } else {
        // Normalize Wasserstein with sigmoid
        normalizedDistance = 2 / (1 + Math.exp(-distanceMetrics.wasserstein * 5)) - 1;
      }
      
      // Calculate resilience as inverse of (impact / distance)
      // Higher distance with lower impact means better resilience
      if (normalizedDistance < 0.01) {
        return 1.0; // Avoid division by very small numbers
      }
      
      const rawResilience = 1 - (performanceImpact.impactScore / normalizedDistance);
      
      // Ensure result is in [0,1] range
      return Math.min(1, Math.max(0, rawResilience));
    }
    
    /**
     * Calculate mean of an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} Mean value
     */
    calculateMean(values) {
      if (!values || values.length === 0) {
        return null;
      }
      
      const sum = values.reduce((acc, val) => acc + val, 0);
      return sum / values.length;
    }
    
    /**
     * Calculate standard deviation of an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} Standard deviation
     */
    calculateStdDev(values) {
      if (!values || values.length <= 1) {
        return null;
      }
      
      const mean = this.calculateMean(values);
      const sumSquaredDiffs = values.reduce((acc, val) => {
        return acc + Math.pow(val - mean, 2);
      }, 0);
      
      return Math.sqrt(sumSquaredDiffs / values.length);
    }
    
    /**
     * Generate synthetic shifted distribution for testing
     * @param {Array} original - Original distribution values
     * @param {string} shiftType - Type of shift to generate
     * @param {number} magnitude - Magnitude of the shift (0-1)
     * @return {Array} Shifted distribution values
     */
    generateShiftedDistribution(original, shiftType, magnitude = 0.5) {
      if (!original || original.length === 0) {
        return [];
      }
      
      // Calculate original statistics
      const mean = this.calculateMean(original);
      const stdDev = this.calculateStdDev(original);
      
      // Generate shifted distribution based on shift type
      switch (shiftType) {
        case 'covariate_shift':
          // Shift mean, keep variance similar
          return original.map(val => val + mean * magnitude);
          
        case 'concept_drift':
          // Keep mean similar, change variance
          return original.map(val => {
            const normalized = (val - mean) / stdDev;
            return mean + normalized * stdDev * (1 + magnitude);
          });
          
        case 'dataset_shift':
          // Change both mean and variance
          return original.map(val => {
            const normalized = (val - mean) / stdDev;
            return (mean + mean * magnitude) + normalized * stdDev * (1 + magnitude);
          });
          
        case 'minor_shift':
          // Small random noise
          return original.map(val => {
            const noise = (Math.random() - 0.5) * stdDev * magnitude;
            return val + noise;
          });
          
        default:
          return [...original]; // Return copy of original if type is unknown
      }
    }
  }
  
  export default ShiftAnalyzer;