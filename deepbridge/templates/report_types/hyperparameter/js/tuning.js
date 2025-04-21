/**
 * JavaScript for rendering hyperparameter tuning order visualizations
 */

// Initialize tuning charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Listen for the renderCharts event
    document.addEventListener('renderCharts', renderTuningOrderVisualization);
});

/**
 * Render tuning order visualization
 */
function renderTuningOrderVisualization() {
    if (!window.reportData) {
        console.error('Cannot render tuning order: reportData not available');
        return;
    }
    
    console.log('Rendering hyperparameter tuning order...');
    
    // Get the container element
    const container = document.getElementById('tuning-order-container');
    if (!container) return;
    
    // Get tuning order data from report data
    const tuningData = window.reportData.tuning_order || [];
    
    if (tuningData.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No tuning order data available.</div>';
        return;
    }
    
    // Create ordered list container
    const orderListContainer = document.createElement('div');
    orderListContainer.className = 'tuning-order-list';
    
    // Create ordered list
    const orderedList = document.createElement('ol');
    orderedList.className = 'ordered-list';
    
    // Add items for each parameter
    tuningData.forEach(item => {
        const listItem = document.createElement('li');
        listItem.className = 'param-item';
        
        // Create parameter card
        const paramCard = document.createElement('div');
        paramCard.className = 'param-card';
        
        // Parameter name
        const paramName = document.createElement('div');
        paramName.className = 'param-name';
        paramName.textContent = item.parameter;
        paramCard.appendChild(paramName);
        
        // Importance score
        const paramScore = document.createElement('div');
        paramScore.className = 'param-score';
        paramScore.textContent = `Importance Score: ${item.score.toFixed(4)}`;
        paramCard.appendChild(paramScore);
        
        // Parameter description
        if (item.description) {
            const paramDesc = document.createElement('div');
            paramDesc.className = 'param-desc';
            paramDesc.textContent = item.description;
            paramCard.appendChild(paramDesc);
        }
        
        // Parameter range
        if (item.range) {
            const paramRange = document.createElement('div');
            paramRange.className = 'param-range';
            
            const rangeLabel = document.createElement('span');
            rangeLabel.className = 'param-range-label';
            rangeLabel.textContent = 'Recommended Range:';
            paramRange.appendChild(rangeLabel);
            
            const rangeValue = document.createElement('span');
            rangeValue.textContent = item.range;
            paramRange.appendChild(rangeValue);
            
            paramCard.appendChild(paramRange);
        }
        
        listItem.appendChild(paramCard);
        orderedList.appendChild(listItem);
    });
    
    orderListContainer.appendChild(orderedList);
    
    // Clear container and add the ordered list
    container.innerHTML = '';
    container.appendChild(orderListContainer);
}