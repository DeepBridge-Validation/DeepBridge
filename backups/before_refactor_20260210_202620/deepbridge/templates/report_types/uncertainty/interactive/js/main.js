// Report initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log("Iniciando carregamento do relatório");
    
    // Initialize main controller
    MainController.init();
    
    // Criar um pequeno atraso para garantir que o DOM esteja completamente carregado
    setTimeout(() => {
        // Initialize section controllers
        console.log("Inicializando controladores de seção");
        OverviewController.init();
        DetailsController.init();
        BoxplotController.init();
        
        // Inicializar os controladores de features
        console.log("Inicializando controladores de features");
        FeatureImportanceController.init();
        
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
    }, 300); // Aumentar o atraso para garantir que todos os scripts estejam carregados
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
        
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                console.log(`Clicou no botão: ${this.textContent}, data-tab=${this.getAttribute('data-tab')}`);
                
                // Remove active class from all buttons
                tabButtons.forEach(btn => btn.classList.remove('active'));
                
                // Add active class to clicked button
                this.classList.add('active');
                
                // Hide all tab contents
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    console.log(`Removendo classe 'active' de ${content.id}`);
                });
                
                // Show target tab content
                const targetTab = this.getAttribute('data-tab');
                const targetElement = document.getElementById(targetTab);
                
                if (targetElement) {
                    console.log(`Adicionando classe 'active' para ${targetTab}`);
                    targetElement.classList.add('active');
                } else {
                    console.error(`Elemento alvo não encontrado: #${targetTab}`);
                }
            });
        });
    }
};