// DistanceMetrics.js - placeholder
/**
 * Distance Metrics
 * 
 * Provides methods to calculate statistical distances between distributions,
 * which are essential for quantifying distribution shifts.
 */

class DistanceMetrics {
    /**
     * Initialize the distance metrics calculator
     */
    constructor() {
      // Configuration for calculations
      this.config = {
        binCount: 20,  // Number of bins for histogram-based calculations
        epsilon: 1e-10  // Small value to avoid division by zero
      };
    }
    
    /**
     * Calculate Kullback-Leibler divergence between two distributions
     * @param {Array} p - First distribution samples
     * @param {Array} q - Second distribution samples
     * @param {number} binCount - Optional number of bins to use
     * @return {number} KL divergence
     */
    calculateKLDivergence(p, q, binCount = this.config.binCount) {
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...p), Math.min(...q));
      const max = Math.max(Math.max(...p), Math.max(...q));
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      p.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      q.forEach(val => {
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
      
      for (let i = 0; i < binCount; i++) {
        if (probP[i] > 0) {
          divergence += probP[i] * Math.log((probP[i] + this.config.epsilon) / 
                                             (probQ[i] + this.config.epsilon));
        }
      }
      
      return divergence;
    }
    
    /**
     * Calculate Jensen-Shannon divergence between two distributions
     * @param {Array} p - First distribution samples
     * @param {Array} q - Second distribution samples
     * @param {number} binCount - Optional number of bins to use
     * @return {number} JS divergence (0-1 range)
     */
    calculateJSDivergence(p, q, binCount = this.config.binCount) {
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...p), Math.min(...q));
      const max = Math.max(Math.max(...p), Math.max(...q));
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      p.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      q.forEach(val => {
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
      
      for (let i = 0; i < binCount; i++) {
        if (probP[i] > 0) {
          klPM += probP[i] * Math.log((probP[i] + this.config.epsilon) / 
                                       (probM[i] + this.config.epsilon));
        }
        
        if (probQ[i] > 0) {
          klQM += probQ[i] * Math.log((probQ[i] + this.config.epsilon) / 
                                       (probM[i] + this.config.epsilon));
        }
      }
      
      // JS = 0.5 * (KL(P||M) + KL(Q||M))
      return 0.5 * (klPM + klQM);
    }
    
    /**
     * Calculate Wasserstein distance (Earth Mover's Distance)
     * @param {Array} p - First distribution samples
     * @param {Array} q - Second distribution samples
     * @return {number} Wasserstein distance
     */
    calculateWassersteinDistance(p, q) {
      // Sort distributions
      const sortedP = [...p].sort((a, b) => a - b);
      const sortedQ = [...q].sort((a, b) => a - b);
      
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
      
      // Calculate 1st Wasserstein distance (L1 norm of CDF difference)
      let distance = 0;
      for (let i = 0; i < minLength; i++) {
        distance += Math.abs(resampledP[i] - resampledQ[i]);
      }
      
      return distance / minLength;
    }
    
    /**
     * Calculate Hellinger distance between two distributions
     * @param {Array} p - First distribution samples
     * @param {Array} q - Second distribution samples
     * @param {number} binCount - Optional number of bins to use
     * @return {number} Hellinger distance (0-1 range)
     */
    calculateHellingerDistance(p, q, binCount = this.config.binCount) {
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...p), Math.min(...q));
      const max = Math.max(Math.max(...p), Math.max(...q));
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      p.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      q.forEach(val => {
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
     * Calculate all distance metrics between two distributions
     * @param {Array} p - First distribution samples
     * @param {Array} q - Second distribution samples
     * @return {Object} All distance metrics
     */
    calculateAll(p, q) {
      if (!p || !q || p.length < 5 || q.length < 5) {
        return {
          kl_divergence: null,
          js_divergence: null,
          wasserstein: null,
          hellinger: null
        };
      }
      
      return {
        kl_divergence: this.calculateKLDivergence(p, q),
        js_divergence: this.calculateJSDivergence(p, q),
        wasserstein: this.calculateWassersteinDistance(p, q),
        hellinger: this.calculateHellingerDistance(p, q)
      };
    }
    
    /**
     * Get the most appropriate distance metric for a given data type
     * @param {Array} p - First distribution samples
     * @param {Array} q - Second distribution samples
     * @param {string} dataType - Type of data ('numerical', 'categorical', etc.)
     * @return {Object} Most appropriate distance metric and value
     */
    getBestMetric(p, q, dataType = 'numerical') {
      const metrics = this.calculateAll(p, q);
      
      // For numerical data, Wasserstein often works well
      if (dataType === 'numerical') {
        return {
          name: 'wasserstein',
          value: metrics.wasserstein,
          normalized: Math.min(1, 2 / (1 + Math.exp(-metrics.wasserstein * 5)) - 1)
        };
      }
      
      // For categorical or probability distributions, JS divergence is often best
      return {
        name: 'js_divergence',
        value: metrics.js_divergence,
        normalized: metrics.js_divergence // JS is already normalized to [0,1]
      };
    }
  }
  
  export default DistanceMetrics;