// Report initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log("Iniciando carregamento do relatório");

    // Initialize main controller
    MainController.init();

    // Aumentar o atraso para garantir que TODOS os scripts estejam carregados
    // Especialmente importante para ResilienceDataManager e ResilienceOverviewChartManager
    setTimeout(() => {
        console.log("Verificando disponibilidade dos componentes...");

        // Verificar se os componentes necessários estão disponíveis
        const componentsReady = {
            ResilienceDataManager: typeof ResilienceDataManager !== 'undefined',
            ResilienceOverviewChartManager: typeof ResilienceOverviewChartManager !== 'undefined',
            OverviewController: typeof OverviewController !== 'undefined',
            DetailsController: typeof DetailsController !== 'undefined',
            BoxplotController: typeof BoxplotController !== 'undefined',
            FeatureImportanceController: typeof FeatureImportanceController !== 'undefined'
        };

        console.log("Componentes disponíveis:", componentsReady);

        // Initialize section controllers
        console.log("Inicializando controladores de seção");
        if (componentsReady.OverviewController) {
            OverviewController.init();
        } else {
            console.error("OverviewController não disponível");
        }

        if (componentsReady.DetailsController) {
            DetailsController.init();
        }

        if (componentsReady.BoxplotController) {
            BoxplotController.init();
        }

        // Inicializar os controladores de features
        console.log("Inicializando controladores de features");
        if (componentsReady.FeatureImportanceController) {
            FeatureImportanceController.init();
        }

        // Inicializar a tabela de features (novo componente)
        if (typeof FeatureImportanceTableController !== 'undefined') {
            console.log("Inicializando tabela de features");
            FeatureImportanceTableController.init();
        } else {
            console.warn("Controlador da tabela de features não encontrado");
        }

        // Forçar resetar as abas para garantir que apenas a primeira esteja ativa
        const tabContents = document.querySelectorAll('.tab-content');
        tabContents.forEach(content => {
            if (content.id === 'overview') {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        console.log("Report initialized with all controllers");
    }, 500); // Aumentado de 300ms para 500ms para garantir carregamento completo
});

// Simple MainController implementation
const MainController = {
    init: function() {
        console.log("MainController initialized");
        this.initTabNavigation();
    },
    
    initTabNavigation: function() {
        const mainTabs = document.getElementById('main-tabs');
        if (!mainTabs) {
            console.error("Não foi possível encontrar o elemento de navegação principal #main-tabs");
            return;
        }
        
        const tabButtons = mainTabs.querySelectorAll('.tab-btn');
        console.log(`Encontrados ${tabButtons.length} botões de navegação`);
        
        const tabContents = document.querySelectorAll('.tab-content');
        console.log(`Encontrados ${tabContents.length} conteúdos de abas`);
        
        // Verificar abas existentes
        tabContents.forEach(content => {
            console.log(`Conteúdo de aba: id=${content.id}, classe=${content.className}`);
        });
        
        // Garantir que apenas a aba inicial esteja ativa
        tabContents.forEach(content => {
            if (content.id === 'overview') {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
        
        // Tab navigation is now handled by the generic tab-navigation.js
        // No need to add duplicate event listeners here
        console.log('Tab navigation handled by tab-navigation.js');
    }
};