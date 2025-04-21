import DataExtractor from './modules/data/DataExtractor.js';
import DataModel from './modules/data/DataModel.js';
import ChartFactory from './modules/charts/ChartFactory.js';
import OverviewController from './controllers/overview/OverviewController.js';
import DetailsController from './controllers/details/DetailsController.js';
import BoxplotController from './controllers/boxplot/BoxplotController.js';
import FeatureController from './controllers/feature/FeatureController.js';

document.addEventListener('DOMContentLoaded', function() {
  // Initialize data services
  const dataExtractor = new DataExtractor();
  const dataModel = new DataModel();
  const chartFactory = new ChartFactory();
  
  // Initialize controllers
  const overviewController = new OverviewController(window.reportData, dataExtractor, chartFactory);
  const detailsController = new DetailsController(window.reportData, dataExtractor, chartFactory);
  const boxplotController = new BoxplotController(window.reportData, dataExtractor, chartFactory);
  const featureController = new FeatureController(window.reportData, dataExtractor, chartFactory);
  
  // Set up main tab navigation
  const mainTabButtons = document.querySelectorAll('.main-tabs .tab-btn');
  const tabContainers = document.querySelectorAll('.tab-content');
  
  mainTabButtons.forEach(button => {
    button.addEventListener('click', function() {
      const tabId = this.getAttribute('data-tab');
      
      // Update active tab button
      mainTabButtons.forEach(btn => btn.classList.remove('active'));
      this.classList.add('active');
      
      // Update visible tab content
      tabContainers.forEach(container => {
        container.classList.toggle('active', container.id === tabId);
      });
      
      // Trigger tab changed event for lazy loading
      document.dispatchEvent(new CustomEvent('tab-changed', {
        detail: { tabId: tabId }
      }));
    });
  });
  
  // Initialize the active tab
  const activeTab = document.querySelector('.main-tabs .tab-btn.active');
  if (activeTab) {
    document.dispatchEvent(new CustomEvent('tab-changed', {
      detail: { tabId: activeTab.getAttribute('data-tab') }
    }));
  }
});