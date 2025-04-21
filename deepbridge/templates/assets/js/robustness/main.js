// Main controller for robustness report
const MainController = {
    init: function() {
        console.log("MainController initialized");
        
        // Initialize tab navigation
        this.initTabNavigation();
        
        // Setup theme switcher if exists
        this.setupThemeSwitcher();
        
        // Log initial data load
        console.log("Report data loaded:", 
            window.reportData ? "Success" : "Failed", 
            "with configuration:",
            window.reportConfig || "None"
        );
    },
    
    initTabNavigation: function() {
        const mainTabs = document.getElementById('main-tabs');
        if (\!mainTabs) return;
        
        const tabButtons = mainTabs.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                tabButtons.forEach(btn => btn.classList.remove('active'));
                
                // Add active class to clicked button
                this.classList.add('active');
                
                // Hide all tab contents
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Show target tab content
                const targetTab = this.getAttribute('data-tab');
                document.getElementById(targetTab).classList.add('active');
                
                // Trigger resize event to fix Plotly charts
                window.dispatchEvent(new Event('resize'));
            });
        });
    },
    
    setupThemeSwitcher: function() {
        const themeSwitcher = document.getElementById('theme-switcher');
        if (\!themeSwitcher) return;
        
        themeSwitcher.addEventListener('change', function() {
            const themeStylesheet = document.getElementById('theme-stylesheet');
            if (\!themeStylesheet) return;
            
            const isDarkMode = this.checked;
            const themePath = isDarkMode 
                ? '/assets/css/robustness/themes/dark.css'
                : '/assets/css/robustness/themes/light.css';
                
            themeStylesheet.setAttribute('href', themePath);
            
            // Update charts if they exist with new theme colors
            if (window.updateChartsTheme) {
                window.updateChartsTheme(isDarkMode);
            }
        });
    }
};
EOL < /dev/null
