/**
 * Tab Navigation System for DeepBridge Reports
 * Handles tab switching with proper display control
 */

(function() {
    'use strict';

    // Wait for DOM to be fully loaded
    function initializeTabNavigation() {
        console.log('[TabNav] Initializing tab navigation system...');

        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        if (tabButtons.length === 0) {
            console.warn('[TabNav] No tab buttons found');
            return;
        }

        if (tabContents.length === 0) {
            console.warn('[TabNav] No tab contents found');
            return;
        }

        console.log(`[TabNav] Found ${tabButtons.length} tabs and ${tabContents.length} content sections`);

        // Function to show a specific tab
        function showTab(tabId) {
            console.log(`[TabNav] Switching to tab: ${tabId}`);

            // Hide all tab contents
            tabContents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
                content.style.opacity = '0';
            });

            // Deactivate all tab buttons
            tabButtons.forEach(btn => {
                btn.classList.remove('active');
            });

            // Show the selected tab content
            const targetContent = document.getElementById(tabId);
            if (targetContent) {
                // Use setTimeout to ensure CSS transition works
                targetContent.style.display = 'block';
                setTimeout(() => {
                    targetContent.classList.add('active');
                    targetContent.style.opacity = '1';
                }, 10);

                // Activate the corresponding button
                const targetButton = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
                if (targetButton) {
                    targetButton.classList.add('active');
                }

                // Trigger custom event
                const event = new CustomEvent('tabchange', {
                    detail: { tabId: tabId, element: targetContent }
                });
                document.dispatchEvent(event);

                // Initialize charts if they exist
                initializeTabCharts(tabId);

                console.log(`[TabNav] Successfully switched to ${tabId}`);
            } else {
                console.error(`[TabNav] Tab content not found: ${tabId}`);
            }
        }

        // Initialize chart functions based on tab
        function initializeTabCharts(tabId) {
            // Map tab IDs to their initialization functions
            const chartInitializers = {
                'overview': 'initializeOverviewCharts',
                'model-comparison': 'initializeComparisonCharts',
                'hyperparameter-analysis': 'initializeHyperparameterCharts',
                'performance-metrics': 'initializePerformanceCharts',
                'tradeoff-analysis': 'initializeTradeoffCharts',
                'ks-statistic': 'initializeKSStatisticCharts',
                'frequency-distribution': 'initializeFrequencyDistributionCharts'
            };

            const initFunction = chartInitializers[tabId];
            if (initFunction && typeof window[initFunction] === 'function') {
                console.log(`[TabNav] Initializing charts for ${tabId}`);
                window[initFunction]();
            }
        }

        // Add click handlers to all tab buttons
        tabButtons.forEach(button => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();

                const tabId = this.getAttribute('data-tab');
                if (tabId) {
                    showTab(tabId);
                } else {
                    console.error('[TabNav] Tab button missing data-tab attribute');
                }
            });
        });

        // Initialize first tab
        const firstButton = tabButtons[0];
        if (firstButton) {
            const firstTabId = firstButton.getAttribute('data-tab');
            if (firstTabId) {
                console.log(`[TabNav] Initializing with first tab: ${firstTabId}`);
                showTab(firstTabId);
            }
        }

        // Handle direct hash navigation
        function handleHashChange() {
            const hash = window.location.hash.slice(1);
            if (hash) {
                const tabExists = document.getElementById(hash);
                if (tabExists && tabExists.classList.contains('tab-content')) {
                    showTab(hash);
                }
            }
        }

        window.addEventListener('hashchange', handleHashChange);
        handleHashChange(); // Check initial hash

        console.log('[TabNav] Tab navigation system initialized successfully');
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeTabNavigation);
    } else {
        // DOM is already loaded
        initializeTabNavigation();
    }

    // Export for global access
    window.TabNavigation = {
        showTab: function(tabId) {
            const event = new Event('DOMContentLoaded');
            initializeTabNavigation();
            setTimeout(() => {
                const tabButtons = document.querySelectorAll('.tab-btn');
                tabButtons.forEach(button => {
                    if (button.getAttribute('data-tab') === tabId) {
                        button.click();
                    }
                });
            }, 100);
        }
    };
})();