"""
Comprehensive robustness testing suite.

This module provides a unified interface for running multiple types
of robustness tests on machine learning models using DBDataset objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import datetime
import os
import json
import time
import warnings
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from .base_wrapper import BaseWrapper
from .feature_perturbation import FeaturePerturbationTests
from .outlier_robustness import OutlierRobustnessTests
from .distribution_shift import DistributionShiftTests
from .adversarial_robustness import AdversarialRobustnessTests


class RobustnessSuite:
    """
    Comprehensive suite for model robustness testing.
    
    This class provides a unified interface for running multiple types
    of robustness tests on machine learning models, including feature
    perturbation, outlier robustness, distribution shift, and adversarial
    attacks.
    """
    
    # Configurações predefinidas
    _CONFIG_TEMPLATES = {
        'quick': {
            'feature_perturbation': [
                {'type': 'noise', 'params': {'level': 0.2}}
            ],
            'outlier_robustness': [
                {'type': 'isolation_forest', 'params': {'contamination': 0.1}}
            ],
            'distribution_shift': [
                {'type': 'mean', 'params': {'levels': [0.2]}}
            ]
        },
        
        'medium': {
            'feature_perturbation': [
                {'type': 'noise', 'params': {'level': 0.2}},
                {'type': 'zero', 'params': {'level': 0.2}},
                {'type': 'all_features', 'params': {'perturbation_type': 'noise', 'level': 0.2}}
            ],
            'outlier_robustness': [
                {'type': 'isolation_forest', 'params': {'contamination': 0.1}},
                {'type': 'analyze', 'params': {}}
            ],
            'distribution_shift': [
                {'type': 'mean', 'params': {'levels': [0.2]}},
                {'type': 'variance', 'params': {'levels': [0.2]}},
                {'type': 'resilience', 'params': {'shift_level': 0.2}}
            ],
            'adversarial_robustness': [
                {'type': 'blackbox_random', 'params': {'epsilon': 0.1}}
            ]
        },
        
        'full': {
            'feature_perturbation': [
                {'type': 'noise', 'params': {'level': 0.2}},
                {'type': 'zero', 'params': {'level': 0.2}},
                {'type': 'quantile', 'params': {'level': 0.2}},
                {'type': 'missing', 'params': {'level': 0.2}},
                {'type': 'salt_pepper', 'params': {'level': 0.1}},
                {'type': 'all_features', 'params': {'perturbation_type': 'noise', 'level': 0.2}}
            ],
            'outlier_robustness': [
                {'type': 'isolation_forest', 'params': {'contamination': 0.1}},
                {'type': 'lof', 'params': {'contamination': 0.1}},
                {'type': 'quantile', 'params': {'contamination': 0.1}},
                {'type': 'analyze', 'params': {}}
            ],
            'distribution_shift': [
                {'type': 'mean', 'params': {'levels': [0.1, 0.2, 0.5]}},
                {'type': 'variance', 'params': {'levels': [0.1, 0.2, 0.5]}},
                {'type': 'skew', 'params': {'levels': [0.1, 0.2, 0.5]}},
                {'type': 'resilience', 'params': {'shift_level': 0.2}},
                {'type': 'batch', 'params': {'n_top_features': 5}}
            ],
            'adversarial_robustness': [
                {'type': 'blackbox_random', 'params': {'epsilon': 0.1}},
                {'type': 'compare', 'params': {'epsilons': [0.01, 0.05, 0.1]}}
            ]
        }
    }
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the robustness testing suite.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        # Resto do código de inicialização permanece o mesmo
        self.dataset = dataset
        self.verbose = verbose
        
        # Inicializar test classes
        if self.verbose:
            print("Initializing feature perturbation tests...")
        self.feature_tests = FeaturePerturbationTests(dataset, verbose)
        
        # ... [resto do código de inicialização]
        
        # Armazenar configuração atual
        self.current_config = None
    
    def config(self, config_name: str) -> 'RobustnessSuite':
        """
        Set a predefined configuration for robustness tests.
        
        Parameters:
        -----------
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
                
        Returns:
        --------
        self : Returns self to allow method chaining
        """
        if config_name not in self._CONFIG_TEMPLATES:
            raise ValueError(f"Unknown configuration: {config_name}. Available options: {list(self._CONFIG_TEMPLATES.keys())}")
                
        self.current_config = self._CONFIG_TEMPLATES[config_name]
        
        if self.verbose:
            print(f"\nConfigured for {config_name} robustness test suite")
            print(f"\nTests that will be executed:")
            
            # Imprimir todos os testes configurados
            for category, tests in self.current_config.items():
                print(f"\n{category.replace('_', ' ').title()}:")
                for i, test in enumerate(tests, 1):
                    test_type = test['type']
                    params = test.get('params', {})
                    param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                    print(f"  {i}. {test_type} ({param_str})")
        
        return self
    
    def run(self) -> Dict[str, Any]:
        """
        Run the currently configured robustness test suite.
        
        Returns:
        --------
        dict : Test results and comprehensive summary with Plotly visualizations
        """
        if self.current_config is None:
            # Default to quick config if none selected
            if self.verbose:
                print("No configuration set, using 'quick' configuration")
            self.config('quick')
                
        if self.verbose:
            print(f"Running robustness test suite...")
            start_time = time.time()
                
        # Run the tests using the current configuration
        results = self.run_custom_test(self.current_config)
        
        # Organizar os resultados em uma estrutura mais acessível
        organized_results = {
            'summary': {
                'overall_score': results['robustness_scores']['overall_score'],
                'category_scores': {k: v for k, v in results['robustness_scores'].items() if k != 'overall_score'},
                'execution_time': None,  # Será preenchido abaixo
                'test_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config_used': self._get_current_config_name()
            },
            'category_results': {},
            'detailed_results': results,
            'visualizations': {}  # Armazenar visualizações Plotly aqui
        }
        
        # Processar os resultados de cada categoria
        for category, tests in results.items():
            if category == 'robustness_scores':
                continue
                
            # Adicionar resumos por categoria
            if category not in organized_results['category_results']:
                organized_results['category_results'][category] = {}
                
            for test_type, test_result in tests.items():
                # Extrair métricas relevantes para cada tipo de teste
                # [código de extração omitido para brevidade]
                pass
        
        # Gerar visualizações em Plotly
        organized_results['visualizations'] = self._generate_plotly_visualizations(results)
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            organized_results['summary']['execution_time'] = elapsed_time
            print(f"Test suite completed in {elapsed_time:.2f} seconds")
            print(f"Overall robustness score: {organized_results['summary']['overall_score']:.3f}")
            print(f"Generated {len(organized_results['visualizations'])} interactive visualizations")
            
        # Adicionar os resultados organizados ao objeto para acesso posterior
        results['results'] = organized_results
                
        return results

    def _generate_plotly_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate Plotly visualizations for test results.
        
        Parameters:
        -----------
        results : dict
            Test results
            
        Returns:
        --------
        dict : Dictionary mapping visualization names to HTML code
        """
        visualizations = {}
        
        # 1. Robustness Summary Radar Chart
        if 'robustness_scores' in results:
            scores = results['robustness_scores']
            
            # Categories to plot (excluding overall score)
            categories = [k for k in scores.keys() if k != 'overall_score']
            values = [scores[cat] for cat in categories]
            
            # Make the plot circular
            categories.append(categories[0])
            values.append(values[0])
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Robustness'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Robustness Summary",
                showlegend=False,
                height=500,
                width=600
            )
            
            # Add overall score annotation
            fig.add_annotation(
                text=f"Overall Score: {scores['overall_score']:.2f}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="darkblue"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            # Store the visualization
            visualizations['robustness_summary'] = pio.to_html(
                fig, 
                full_html=False, 
                config={'displayModeBar': False}
            )
        
        # 2. Feature Importance Bar Chart
        if 'feature_perturbation' in results:
            for test_type, test_result in results['feature_perturbation'].items():
                if test_type == 'all_features' and 'feature_importance' in test_result:
                    # Extract feature importance
                    importance = test_result['feature_importance']
                    
                    # Sort features by importance
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Limit to top 10 features
                    if len(sorted_features) > 10:
                        sorted_features = sorted_features[:10]
                    
                    features = [f[0] for f in sorted_features]
                    importance_values = [f[1] for f in sorted_features]
                    
                    # Create bar chart
                    fig = go.Figure(go.Bar(
                        x=importance_values,
                        y=features,
                        orientation='h',
                        marker=dict(
                            color=importance_values,
                            colorscale='Viridis',
                            colorbar=dict(title='Importance')
                        )
                    ))
                    
                    fig.update_layout(
                        title="Feature Importance Based on Robustness",
                        xaxis_title="Robustness Importance",
                        yaxis=dict(
                            title="Feature",
                            autorange="reversed"  # Labels read top-to-bottom
                        ),
                        height=500,
                        width=700
                    )
                    
                    # Store the visualization
                    visualizations['feature_importance'] = pio.to_html(
                        fig, 
                        full_html=False, 
                        config={'displayModeBar': False}
                    )
                    
                    break
        
        # 3. Outlier Performance Comparison
        if 'outlier_robustness' in results:
            for test_type, test_result in results['outlier_robustness'].items():
                if test_type != 'analysis' and 'outlier_performance' in test_result and 'inlier_performance' in test_result:
                    # Get the most relevant metric
                    if 'accuracy' in test_result['outlier_performance']:
                        metric = 'accuracy'
                    elif 'mse' in test_result['outlier_performance']:
                        metric = 'mse'
                    else:
                        # Use the first available metric
                        metric = list(test_result['outlier_performance'].keys())[0]
                    
                    # Extract values
                    outlier_value = test_result['outlier_performance'][metric]
                    inlier_value = test_result['inlier_performance'][metric]
                    
                    # Create comparison bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Inliers', 'Outliers'],
                            y=[inlier_value, outlier_value],
                            marker_color=['blue', 'red']
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"Model Performance: Outliers vs. Inliers ({test_type})",
                        yaxis_title=f"{metric.upper()} Performance",
                        height=400,
                        width=600
                    )
                    
                    # Add relative change annotation
                    rel_change = (outlier_value - inlier_value) / inlier_value if inlier_value != 0 else 0
                    fig.add_annotation(
                        text=f"Relative Change: {rel_change:.2%}",
                        x=0.5, y=1.05,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=14)
                    )
                    
                    # Store the visualization
                    visualizations[f'outlier_comparison_{test_type}'] = pio.to_html(
                        fig, 
                        full_html=False, 
                        config={'displayModeBar': False}
                    )
                    
                    break
        
        # 4. Distribution Shift Impact
        if 'distribution_shift' in results:
            for test_type, test_result in results['distribution_shift'].items():
                if test_type in ['mean', 'variance', 'skew'] and 'performance' in test_result and len(test_result['performance']) > 0:
                    # Extract performance data
                    levels = [p['level'] for p in test_result['performance']]
                    
                    # Get the most relevant metric
                    if 'accuracy' in test_result['performance'][0]['relative_change']:
                        metric = 'accuracy'
                    elif 'mse' in test_result['performance'][0]['relative_change']:
                        metric = 'mse'
                    else:
                        # Use the first available metric
                        metric = list(test_result['performance'][0]['relative_change'].keys())[0]
                    
                    rel_changes = [p['relative_change'][metric] for p in test_result['performance']]
                    
                    # Create line chart
                    fig = go.Figure(data=go.Scatter(
                        x=levels,
                        y=rel_changes,
                        mode='lines+markers',
                        marker=dict(size=10),
                        line=dict(width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"Impact of {test_type.title()} Shift on {metric.upper()}",
                        xaxis_title=f"{test_type.title()} Shift Level",
                        yaxis_title=f"Relative Change in {metric.upper()}",
                        height=400,
                        width=600
                    )
                    
                    # Add reference line at y=0
                    fig.add_shape(
                        type="line",
                        x0=min(levels),
                        y0=0,
                        x1=max(levels),
                        y1=0,
                        line=dict(color="gray", width=2, dash="dash")
                    )
                    
                    # Store the visualization
                    visualizations[f'distribution_shift_{test_type}'] = pio.to_html(
                        fig, 
                        full_html=False, 
                        config={'displayModeBar': False}
                    )
        
        # 5. Adversarial Attack Success Rate
        if 'adversarial_robustness' in results:
            for test_type, test_result in results['adversarial_robustness'].items():
                if test_type == 'compare' and 'success_rates' in test_result:
                    # Extract success rates
                    epsilons = test_result['epsilons']
                    success_rates = {}
                    
                    for attack_type, rates in test_result['success_rates'].items():
                        success_rates[attack_type] = rates
                    
                    # Create line chart
                    fig = go.Figure()
                    
                    for attack_type, rates in success_rates.items():
                        fig.add_trace(go.Scatter(
                            x=epsilons,
                            y=rates,
                            mode='lines+markers',
                            name=attack_type
                        ))
                    
                    fig.update_layout(
                        title="Adversarial Attack Success Rate vs. Strength",
                        xaxis_title="Epsilon (attack strength)",
                        yaxis_title="Attack Success Rate",
                        height=400,
                        width=700,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        )
                    )
                    
                    # Store the visualization
                    visualizations['adversarial_success_rates'] = pio.to_html(
                        fig, 
                        full_html=False, 
                        config={'displayModeBar': False}
                    )
                    
                    break
        
        return visualizations
    


    def save_report(self, output_path: str) -> None:
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("Nenhum resultado disponível. Execute um teste primeiro.")
        
        # Obter o último resultado de teste
        ultima_chave_teste = list(self.results.keys())[-1]
        resultados_teste = self.results[ultima_chave_teste]
        
        # Verificar se precisamos extrair resultados de estrutura aninhada
        if 'robustness_scores' not in resultados_teste and 'results' in resultados_teste:
            # Extrair os resultados organizados
            resultados_organizados = resultados_teste['results']
        else:
            # Organizar resultados agora
            resultados_organizados = {
                'summary': {
                    'overall_score': resultados_teste.get('robustness_scores', {}).get('overall_score', 0),
                    'category_scores': {k: v for k, v in resultados_teste.get('robustness_scores', {}).items() 
                                    if k != 'overall_score'},
                    'test_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'config_used': self._get_current_config_name()
                },
                'detailed_results': resultados_teste,
                'visualizations': self._generate_plotly_visualizations(resultados_teste)
            }
        
        # Carregar o template HTML
        try:
            # Primeiro tenta o caminho original
            template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "visualization", "templates", "robustness_report_template.html"
            )
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            # Tenta um caminho alternativo
            try:
                template_path = "robustness_report_template.html"  # Tenta no diretório atual
                with open(template_path, 'r', encoding='utf-8') as f:
                    template = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("Arquivo de template não encontrado.")
        
        # Preencher o template com os dados
        report_html = self._fill_report_template(template, resultados_organizados)
        
        # Salvar o arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
            
        if self.verbose:
            print(f"Relatório salvo em {output_path}")

    def _fill_report_template(self, template: str, results: Dict[str, Any]) -> str:
        """
        Fill the HTML template with results data.
        
        Parameters:
        -----------
        template : str
            HTML template string
        results : dict
            Organized test results
            
        Returns:
        --------
        str : Filled HTML template
        """
        # Obter informações de resumo
        summary = results.get('summary', {})
        overall_score = summary.get('overall_score', 0)
        category_scores = summary.get('category_scores', {})
        execution_time = summary.get('execution_time', 0)
        test_date = summary.get('test_date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        config_used = summary.get('config_used', 'custom')
        
        # Formatar score com cores baseadas no valor
        def get_score_color(score):
            if score >= 0.8:
                return "#4CAF50"  # Verde para alta robustez
            elif score >= 0.6:
                return "#FFC107"  # Amarelo para robustez média
            else:
                return "#F44336"  # Vermelho para baixa robustez
        
        # Obter informações do modelo e dataset
        model_type = self.model.__class__.__name__ if hasattr(self, 'model') else "Unknown"
        problem_type = self.feature_tests.get_problem_type() if hasattr(self, 'feature_tests') else "Unknown"
        dataset_size = len(self.dataset.test_data) if hasattr(self, 'dataset') else 0
        num_features = len(self.dataset.features) if hasattr(self, 'dataset') else 0
        
        # Formatar scores de categorias
        category_scores_html = ""
        for category, score in category_scores.items():
            color = get_score_color(score)
            category_name = category.replace('_', ' ').title()
            category_scores_html += f'<span class="category-badge" style="background-color: {color}">{category_name}: {score:.2f}</span>'
        
        # Formatar visualizações
        visualizations = results.get('visualizations', {})
        
        # Criar tabs para visualizações
        visualization_tabs = ""
        visualization_content = ""
        
        for i, (viz_name, viz_html) in enumerate(visualizations.items()):
            tab_name = viz_name.replace('_', ' ').title()
            tab_id = f"viz-{viz_name}"
            
            # Adicionar tab
            active_class = " active" if i == 0 else ""
            visualization_tabs += f'<div class="tab{active_class}" onclick="openTab(event, \'{tab_id}\')">{tab_name}</div>'
            
            # Adicionar conteúdo do tab
            visualization_content += f'<div id="{tab_id}" class="tab-content{active_class}"><div class="visualization-container">{viz_html}</div></div>'
        
        # Formatar resultados detalhados para cada categoria
        feature_results_html = self._format_feature_results(results)
        outlier_results_html = self._format_outlier_results(results)
        distribution_results_html = self._format_distribution_results(results)
        adversarial_results_html = self._format_adversarial_results(results)
        
        # Substituir placeholders no template
        filled_template = template.replace("{{overall_score}}", f"{overall_score:.2f}")
        filled_template = filled_template.replace("{{overall_score_color}}", get_score_color(overall_score))
        filled_template = filled_template.replace("{{config_used}}", config_used)
        filled_template = filled_template.replace("{{execution_time}}", f"{execution_time:.2f}")
        filled_template = filled_template.replace("{{test_date}}", test_date)
        filled_template = filled_template.replace("{{category_scores_html}}", category_scores_html)
        filled_template = filled_template.replace("{{model_type}}", model_type)
        filled_template = filled_template.replace("{{problem_type}}", problem_type)
        filled_template = filled_template.replace("{{dataset_size}}", str(dataset_size))
        filled_template = filled_template.replace("{{num_features}}", str(num_features))
        filled_template = filled_template.replace("{{visualization_tabs}}", visualization_tabs)
        filled_template = filled_template.replace("{{visualization_content}}", visualization_content)
        filled_template = filled_template.replace("{{feature_results_html}}", feature_results_html)
        filled_template = filled_template.replace("{{outlier_results_html}}", outlier_results_html)
        filled_template = filled_template.replace("{{distribution_results_html}}", distribution_results_html)
        filled_template = filled_template.replace("{{adversarial_results_html}}", adversarial_results_html)
        filled_template = filled_template.replace("{{current_year}}", str(datetime.datetime.now().year))
        
        return filled_template

    def _format_feature_results(self, results: Dict[str, Any]) -> str:
        """Format feature perturbation results as HTML"""
        if 'category_results' not in results or 'feature_perturbation' not in results['category_results']:
            return "<p>No feature perturbation tests were performed.</p>"
        
        feature_results = results['category_results']['feature_perturbation']
        
        html = "<h3>Feature Perturbation Tests</h3>"
        
        # All Features Test
        if 'all_features' in feature_results:
            html += "<h4>Feature Importance</h4>"
            html += "<table class='result-table'>"
            html += "<tr><th>Feature</th><th>Importance Score</th></tr>"
            
            top_features = feature_results['all_features'].get('top_features', {})
            for feature, score in top_features.items():
                html += f"<tr><td>{feature}</td><td>{score:.4f}</td></tr>"
                
            html += "</table>"
        
        # Individual Feature Tests
        other_tests = {k: v for k, v in feature_results.items() if k != 'all_features'}
        if other_tests:
            html += "<h4>Individual Feature Perturbation Results</h4>"
            html += "<table class='result-table'>"
            html += "<tr><th>Test Type</th><th>Feature</th><th>Relative Performance Change</th></tr>"
            
            for test_type, test_data in other_tests.items():
                feature = test_data.get('feature', 'N/A')
                changes = test_data.get('relative_performance_change', {})
                
                for metric, change in changes.items():
                    color = "red" if change < 0 else "green"
                    html += f"<tr><td>{test_type}</td><td>{feature}</td>"
                    html += f"<td style='color: {color}'>{change:.2%}</td></tr>"
                    
            html += "</table>"
        
        return html

    def _format_outlier_results(self, results: Dict[str, Any]) -> str:
        """Format outlier robustness results as HTML"""
        if 'category_results' not in results or 'outlier_robustness' not in results['category_results']:
            return "<p>No outlier robustness tests were performed.</p>"
        
        outlier_results = results['category_results']['outlier_robustness']
        
        html = "<h3>Outlier Robustness Tests</h3>"
        html += "<table class='result-table'>"
        html += "<tr><th>Detection Method</th><th>Outliers</th><th>Inliers</th>"
        html += "<th>Outlier Performance</th><th>Inlier Performance</th></tr>"
        
        for test_type, test_data in outlier_results.items():
            outlier_perf = test_data.get('outlier_performance', {})
            inlier_perf = test_data.get('inlier_performance', {})
            n_outliers = test_data.get('n_outliers', 0)
            n_inliers = test_data.get('n_inliers', 0)
            
            # Get primary metric
            if 'accuracy' in outlier_perf:
                metric = 'accuracy'
            elif 'mse' in outlier_perf:
                metric = 'mse'
            else:
                metric = list(outlier_perf.keys())[0] if outlier_perf else 'N/A'
            
            outlier_value = outlier_perf.get(metric, 0)
            inlier_value = inlier_perf.get(metric, 0)
            
            rel_change = ((outlier_value - inlier_value) / inlier_value) if inlier_value != 0 else 0
            color = "red" if rel_change < 0 else "green"
            
            html += f"<tr><td>{test_type}</td><td>{n_outliers}</td><td>{n_inliers}</td>"
            html += f"<td>{outlier_value:.4f}</td><td>{inlier_value:.4f}</td></tr>"
            
        html += "</table>"
        return html

    def _format_distribution_results(self, results: Dict[str, Any]) -> str:
        """Format distribution shift results as HTML"""
        if 'category_results' not in results or 'distribution_shift' not in results['category_results']:
            return "<p>No distribution shift tests were performed.</p>"
        
        dist_results = results['category_results']['distribution_shift']
        
        html = "<h3>Distribution Shift Tests</h3>"
        
        # Resilience Score
        if 'resilience' in dist_results:
            resilience = dist_results['resilience']
            html += "<h4>Resilience Scores</h4>"
            html += "<table class='result-table'>"
            html += "<tr><th>Score Type</th><th>Value</th></tr>"
            
            norm_score = resilience.get('normalized_score', 0)
            color = "red" if norm_score < 0.6 else ("orange" if norm_score < 0.8 else "green")
            
            html += f"<tr><td>Normalized Resilience Score</td>"
            html += f"<td style='color: {color}'>{norm_score:.4f}</td></tr>"
            
            feature_scores = resilience.get('feature_scores', {})
            if feature_scores:
                html += "<tr><th colspan='2'>Feature Resilience Scores</th></tr>"
                
                for feature, score in feature_scores.items():
                    feature_color = "red" if score < 0.6 else ("orange" if score < 0.8 else "green")
                    html += f"<tr><td>{feature}</td>"
                    html += f"<td style='color: {feature_color}'>{score:.4f}</td></tr>"
                    
            html += "</table>"
        
        # Distribution Shift Tests
        shift_types = {k: v for k, v in dist_results.items() if k != 'resilience' and k != 'batch'}
        if shift_types:
            html += "<h4>Distribution Shift Results</h4>"
            html += "<table class='result-table'>"
            html += "<tr><th>Shift Type</th><th>Feature</th><th>Relative Performance Change</th></tr>"
            
            for shift_type, shift_data in shift_types.items():
                changes = shift_data.get('relative_performance_change', {})
                feature = shift_data.get('feature', 'N/A')
                
                for metric, change in changes.items():
                    color = "red" if change < 0 else "green"
                    html += f"<tr><td>{shift_type}</td><td>{feature}</td>"
                    html += f"<td style='color: {color}'>{change:.2%}</td></tr>"
                    
            html += "</table>"
        
        return html

    def _format_adversarial_results(self, results: Dict[str, Any]) -> str:
        """Format adversarial robustness results as HTML"""
        if 'category_results' not in results or 'adversarial_robustness' not in results['category_results']:
            return "<p>No adversarial robustness tests were performed.</p>"
        
        adv_results = results['category_results']['adversarial_robustness']
        
        html = "<h3>Adversarial Robustness Tests</h3>"
        html += "<table class='result-table'>"
        html += "<tr><th>Attack Type</th><th>Success Rate</th>"
        html += "<th>Adversarial Performance</th><th>Baseline Performance</th></tr>"
        
        for attack_type, attack_data in adv_results.items():
            adv_perf = attack_data.get('adversarial_performance', {})
            base_perf = attack_data.get('baseline_performance', {})
            success_rate = attack_data.get('success_rate', 0)
            
            # Get primary metric
            if 'accuracy' in adv_perf:
                metric = 'accuracy'
            elif 'mse' in adv_perf:
                metric = 'mse'
            else:
                metric = list(adv_perf.keys())[0] if adv_perf else 'N/A'
            
            adv_value = adv_perf.get(metric, 0)
            base_value = base_perf.get(metric, 0)
            
            color = "red" if success_rate > 0.3 else ("orange" if success_rate > 0.1 else "green")
            
            html += f"<tr><td>{attack_type}</td>"
            html += f"<td style='color: {color}'>{success_rate:.2%}</td>"
            html += f"<td>{adv_value:.4f}</td><td>{base_value:.4f}</td></tr>"
            
        html += "</table>"
        return html
        
    def _get_current_config_name(self):
        """
        Get the name of the current configuration.
        """
        for name, config in self._CONFIG_TEMPLATES.items():
            if config == self.current_config:
                return name
        return "custom"
    
    def __init__(self, dataset, verbose: bool = False):
        """
        Initialize the robustness testing suite.
        
        Parameters:
        -----------
        dataset : DBDataset
            Dataset object containing training/test data and model
        verbose : bool
            Whether to print progress information
        """
        self.dataset = dataset
        self.verbose = verbose
        
        # Initialize all test classes
        if self.verbose:
            print("Initializing feature perturbation tests...")
        self.feature_tests = FeaturePerturbationTests(dataset, verbose)
        
        if self.verbose:
            print("Initializing outlier robustness tests...")
        self.outlier_tests = OutlierRobustnessTests(dataset, verbose)
        
        if self.verbose:
            print("Initializing distribution shift tests...")
        self.distribution_tests = DistributionShiftTests(dataset, verbose)
        
        if self.verbose:
            print("Initializing adversarial robustness tests...")
        try:
            self.adversarial_tests = AdversarialRobustnessTests(dataset, verbose)
            self.adversarial_available = True
        except Exception as e:
            if self.verbose:
                print(f"Adversarial tests not available: {str(e)}")
            self.adversarial_available = False
        
        # Store results
        self.results = {}
    
    def run_quick_test(self) -> Dict[str, Any]:
        """
        Run a quick robustness test with minimal configuration.
        
        This method runs a small subset of tests to quickly assess
        model robustness.
        
        Returns:
        --------
        dict : Quick test results
        """
        if self.verbose:
            print("Running quick robustness test...")
            start_time = time.time()
        
        results = {
            'feature_perturbation': {},
            'outlier_robustness': {},
            'distribution_shift': {}
        }
        
        # Feature perturbation tests
        if self.verbose:
            print("Testing feature perturbation (noise)...")
        results['feature_perturbation']['noise'] = self.feature_tests.test_noise()
        
        # Outlier robustness test
        if self.verbose:
            print("Testing outlier robustness...")
        results['outlier_robustness']['isolation_forest'] = self.outlier_tests.test_isolation_forest()
        
        # Distribution shift test
        if self.verbose:
            print("Testing distribution shift...")
        results['distribution_shift']['mean'] = self.distribution_tests.test_mean_shift()
        
        # Adversarial robustness if available
        if self.adversarial_available:
            if self.verbose:
                print("Testing adversarial robustness...")
            results['adversarial_robustness'] = {}
            results['adversarial_robustness']['blackbox'] = self.adversarial_tests.test_blackbox_random()
        
        # Calculate overall robustness score
        scores = self._calculate_overall_score(results)
        results['robustness_scores'] = scores
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"Quick test completed in {elapsed_time:.2f} seconds")
            print(f"Overall robustness score: {scores['overall_score']:.3f}")
        
        # Store results
        self.results['quick_test'] = results
        
        return results
    
    def run_comprehensive_test(self, perturbation_level: float = 0.2) -> Dict[str, Any]:
        """
        Run a comprehensive robustness test suite.
        
        This method runs a full set of robustness tests to thoroughly
        assess model robustness.
        
        Parameters:
        -----------
        perturbation_level : float
            Level of perturbation to apply (0-1)
            
        Returns:
        --------
        dict : Comprehensive test results
        """
        if self.verbose:
            print("Running comprehensive robustness test...")
            start_time = time.time()
        
        results = {
            'feature_perturbation': {},
            'outlier_robustness': {},
            'distribution_shift': {},
            'adversarial_robustness': {}
        }
        
        # Feature perturbation tests
        if self.verbose:
            print("Testing feature perturbation...")
            
        # Run different types of perturbation
        results['feature_perturbation']['noise'] = self.feature_tests.test_noise(level=perturbation_level)
        results['feature_perturbation']['zero'] = self.feature_tests.test_zero(level=perturbation_level)
        results['feature_perturbation']['quantile'] = self.feature_tests.test_quantile(level=perturbation_level)
        results['feature_perturbation']['missing'] = self.feature_tests.test_missing(level=perturbation_level)
        results['feature_perturbation']['salt_pepper'] = self.feature_tests.test_salt_pepper(level=perturbation_level)
        
        # Test all features
        if self.verbose:
            print("Testing all features...")
        results['feature_perturbation']['all_features'] = self.feature_tests.test_all_features(
            perturbation_type='noise',
            level=perturbation_level
        )
        
        # Outlier robustness tests
        if self.verbose:
            print("Testing outlier robustness...")
            
        # Run different outlier detection methods
        results['outlier_robustness']['isolation_forest'] = self.outlier_tests.test_isolation_forest(
            contamination=0.1
        )
        results['outlier_robustness']['lof'] = self.outlier_tests.test_lof(
            contamination=0.1
        )
        results['outlier_robustness']['quantile'] = self.outlier_tests.test_quantile(
            contamination=0.1
        )
        
        # Analyze outliers
        if self.verbose:
            print("Analyzing outliers...")
        results['outlier_robustness']['analysis'] = self.outlier_tests.analyze_outliers()
        
        # Distribution shift tests
        if self.verbose:
            print("Testing distribution shift...")
            
        # Run different types of distribution shift
        results['distribution_shift']['mean'] = self.distribution_tests.test_mean_shift(
            levels=[perturbation_level]
        )
        results['distribution_shift']['variance'] = self.distribution_tests.test_variance_shift(
            levels=[perturbation_level]
        )
        results['distribution_shift']['skew'] = self.distribution_tests.test_skew_shift(
            levels=[perturbation_level]
        )
        
        # Compute resilience score
        if self.verbose:
            print("Computing resilience score...")
        results['distribution_shift']['resilience'] = self.distribution_tests.compute_resilience_score(
            shift_level=perturbation_level
        )
        
        # Adversarial robustness tests if available
        if self.adversarial_available:
            if self.verbose:
                print("Testing adversarial robustness...")
                
            # Run blackbox attack (works for all models)
            results['adversarial_robustness']['blackbox_random'] = self.adversarial_tests.test_blackbox_random(
                epsilon=perturbation_level
            )
            
            # Check if model supports gradients
            if hasattr(self.adversarial_tests, 'supports_gradients') and self.adversarial_tests.supports_gradients:
                # Run gradient-based attacks
                if self.verbose:
                    print("Running gradient-based attacks...")
                    
                results['adversarial_robustness']['fgsm'] = self.adversarial_tests.test_fgsm(
                    epsilon=perturbation_level
                )
                
                results['adversarial_robustness']['pgd'] = self.adversarial_tests.test_pgd(
                    epsilon=perturbation_level,
                    alpha=perturbation_level/10,
                    iterations=10
                )
        
        # Calculate overall robustness score
        scores = self._calculate_overall_score(results)
        results['robustness_scores'] = scores
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"Comprehensive test completed in {elapsed_time:.2f} seconds")
            print(f"Overall robustness score: {scores['overall_score']:.3f}")
        
        # Store results
        self.results['comprehensive_test'] = results
        
        return results
    
    def run_custom_test(self, tests: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run a custom set of robustness tests.
        
        This method allows specifying exactly which tests to run.
        
        Parameters:
        -----------
        tests : dict
            Dictionary specifying tests to run, e.g.:
            {
                'feature_perturbation': [
                    {'type': 'noise', 'params': {'feature_name': 'f1', 'level': 0.2}},
                    {'type': 'zero', 'params': {'level': 0.3}}
                ],
                'outlier_robustness': [
                    {'type': 'isolation_forest', 'params': {'contamination': 0.1}}
                ]
            }
            
        Returns:
        --------
        dict : Custom test results
        """
        if self.verbose:
            print("Running custom robustness test...")
            start_time = time.time()
        
        results = {}
        
        # Process each test category
        for category, test_list in tests.items():
            if category not in results:
                results[category] = {}
                
            # Run tests for this category
            for test_spec in test_list:
                test_type = test_spec['type']
                params = test_spec.get('params', {})
                
                if self.verbose:
                    print(f"Running {category} test: {test_type}")
                    
                # Run the specified test
                if category == 'feature_perturbation':
                    if test_type == 'noise':
                        result = self.feature_tests.test_noise(**params)
                    elif test_type == 'zero':
                        result = self.feature_tests.test_zero(**params)
                    elif test_type == 'flip':
                        result = self.feature_tests.test_flip(**params)
                    elif test_type == 'quantile':
                        result = self.feature_tests.test_quantile(**params)
                    elif test_type == 'missing':
                        result = self.feature_tests.test_missing(**params)
                    elif test_type == 'salt_pepper':
                        result = self.feature_tests.test_salt_pepper(**params)
                    elif test_type == 'all_features':
                        result = self.feature_tests.test_all_features(**params)
                    else:
                        warnings.warn(f"Unknown feature perturbation test: {test_type}")
                        continue
                        
                elif category == 'outlier_robustness':
                    if test_type == 'isolation_forest':
                        result = self.outlier_tests.test_isolation_forest(**params)
                    elif test_type == 'lof':
                        result = self.outlier_tests.test_lof(**params)
                    elif test_type == 'quantile':
                        result = self.outlier_tests.test_quantile(**params)
                    elif test_type == 'analyze':
                        result = self.outlier_tests.analyze_outliers(**params)
                    else:
                        warnings.warn(f"Unknown outlier robustness test: {test_type}")
                        continue
                        
                elif category == 'distribution_shift':
                    if test_type == 'mean':
                        result = self.distribution_tests.test_mean_shift(**params)
                    elif test_type == 'variance':
                        result = self.distribution_tests.test_variance_shift(**params)
                    elif test_type == 'skew':
                        result = self.distribution_tests.test_skew_shift(**params)
                    elif test_type == 'resilience':
                        result = self.distribution_tests.compute_resilience_score(**params)
                    elif test_type == 'batch':
                        result = self.distribution_tests.batch_test_features(**params)
                    else:
                        warnings.warn(f"Unknown distribution shift test: {test_type}")
                        continue
                        
                elif category == 'adversarial_robustness':
                    if not self.adversarial_available:
                        warnings.warn("Adversarial robustness tests not available")
                        continue
                        
                    if test_type == 'fgsm':
                        result = self.adversarial_tests.test_fgsm(**params)
                    elif test_type == 'pgd':
                        result = self.adversarial_tests.test_pgd(**params)
                    elif test_type == 'blackbox_random':
                        result = self.adversarial_tests.test_blackbox_random(**params)
                    elif test_type == 'blackbox_boundary':
                        result = self.adversarial_tests.test_blackbox_boundary(**params)
                    elif test_type == 'compare':
                        result = self.adversarial_tests.compare_attacks(**params)
                    else:
                        warnings.warn(f"Unknown adversarial robustness test: {test_type}")
                        continue
                else:
                    warnings.warn(f"Unknown test category: {category}")
                    continue
                
                # Store result
                results[category][test_type] = result
        
        # Calculate overall robustness score
        scores = self._calculate_overall_score(results)
        results['robustness_scores'] = scores
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"Custom test completed in {elapsed_time:.2f} seconds")
            print(f"Overall robustness score: {scores['overall_score']:.3f}")
        
        # Store results
        test_id = f"custom_test_{int(time.time())}"
        self.results[test_id] = results
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate overall robustness score from test results.
        
        Parameters:
        -----------
        results : dict
            Test results from various robustness tests
            
        Returns:
        --------
        dict : Robustness scores for different aspects and overall
        """
        scores = {}
        
        # Feature perturbation score
        if 'feature_perturbation' in results:
            feature_scores = []
            
            for test_type, test_result in results['feature_perturbation'].items():
                if test_type == 'all_features' and 'feature_importance' in test_result:
                    # Already have feature importance
                    continue
                    
                # Get primary metric
                if self.feature_tests.is_classification():
                    metric = 'accuracy'
                else:
                    metric = 'mse'
                
                # Calculate score based on relative change
                if 'performance' in test_result and len(test_result['performance']) > 0:
                    rel_change = test_result['performance'][0]['relative_change'].get(metric, 0)
                    
                    # Convert to score (higher is better)
                    score = max(0, 1 + rel_change)
                    feature_scores.append(score)
            
            if feature_scores:
                scores['feature_perturbation'] = np.mean(feature_scores)
        
        # Outlier robustness score
        if 'outlier_robustness' in results:
            outlier_scores = []
            
            for test_type, test_result in results['outlier_robustness'].items():
                if test_type == 'analysis':
                    continue
                    
                # Get primary metric
                if self.outlier_tests.is_classification():
                    metric = 'accuracy'
                else:
                    metric = 'mse'
                
                # Calculate score based on ratio of outlier to inlier performance
                if 'outlier_performance' in test_result and 'inlier_performance' in test_result:
                    outlier_perf = test_result['outlier_performance'].get(metric, 0)
                    inlier_perf = test_result['inlier_performance'].get(metric, 1)
                    
                    if inlier_perf != 0:
                        if metric in ['mse', 'rmse', 'mae']:
                            # For metrics where lower is better
                            ratio = inlier_perf / outlier_perf if outlier_perf != 0 else 0
                        else:
                            # For metrics where higher is better
                            ratio = outlier_perf / inlier_perf
                            
                        # Convert to score (higher is better)
                        score = max(0, min(1, ratio))
                        outlier_scores.append(score)
            
            if outlier_scores:
                scores['outlier_robustness'] = np.mean(outlier_scores)
        
        # Distribution shift score
        if 'distribution_shift' in results:
            if 'resilience' in results['distribution_shift']:
                # Use computed resilience score if available
                resilience = results['distribution_shift']['resilience']
                if 'normalized_score' in resilience:
                    scores['distribution_shift'] = resilience['normalized_score']
            else:
                # Calculate from other tests
                shift_scores = []
                
                for test_type, test_result in results['distribution_shift'].items():
                    if test_type in ['resilience', 'batch']:
                        continue
                        
                    # Get primary metric
                    if self.distribution_tests.is_classification():
                        metric = 'accuracy'
                    else:
                        metric = 'mse'
                    
                    # Calculate score based on relative change
                    if 'performance' in test_result and len(test_result['performance']) > 0:
                        rel_change = test_result['performance'][0]['relative_change'].get(metric, 0)
                        
                        # Convert to score (higher is better)
                        score = max(0, 1 + rel_change)
                        shift_scores.append(score)
                
                if shift_scores:
                    scores['distribution_shift'] = np.mean(shift_scores)
        
        # Adversarial robustness score
        if 'adversarial_robustness' in results and self.adversarial_available:
            adv_scores = []
            
            for test_type, test_result in results['adversarial_robustness'].items():
                if test_type == 'compare':
                    continue
                    
                # Get primary metric
                if self.adversarial_tests.is_classification():
                    metric = 'accuracy'
                else:
                    metric = 'mse'
                
                # Calculate score based on performance ratio
                if 'performance' in test_result and 'baseline_performance' in test_result:
                    adv_perf = test_result['performance'].get(metric, 0)
                    baseline_perf = test_result['baseline_performance'].get(metric, 1)
                    
                    if baseline_perf != 0:
                        if metric in ['mse', 'rmse', 'mae']:
                            # For metrics where lower is better
                            ratio = baseline_perf / adv_perf if adv_perf != 0 else 0
                        else:
                            # For metrics where higher is better
                            ratio = adv_perf / baseline_perf
                            
                        # Convert to score (higher is better)
                        score = max(0, min(1, ratio))
                        adv_scores.append(score)
            
            if adv_scores:
                scores['adversarial_robustness'] = np.mean(adv_scores)
        
        # Calculate overall score
        category_scores = list(scores.values())
        
        if category_scores:
            # Weight the scores (can be customized)
            weights = {
                'feature_perturbation': 0.35,
                'outlier_robustness': 0.25,
                'distribution_shift': 0.25,
                'adversarial_robustness': 0.15
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for category, score in scores.items():
                weight = weights.get(category, 1.0)
                weighted_sum += score * weight
                total_weight += weight
                
            if total_weight > 0:
                overall_score = weighted_sum / total_weight
            else:
                overall_score = np.mean(category_scores)
                
            scores['overall_score'] = overall_score
        else:
            scores['overall_score'] = 0.5  # Default score if no tests were run
        
        return scores
    
    def plot_robustness_summary(self, test_id: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot summary of robustness test results.
        
        Parameters:
        -----------
        test_id : str or None
            ID of test to plot (None for most recent)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Summary plot
        """
        # Get results for the specified test
        if test_id is None:
            if 'comprehensive_test' in self.results:
                test_id = 'comprehensive_test'
            elif 'quick_test' in self.results:
                test_id = 'quick_test'
            elif self.results:
                test_id = list(self.results.keys())[-1]
            else:
                raise ValueError("No test results available")
                
        if test_id not in self.results:
            raise ValueError(f"Test results not found for ID: {test_id}")
            
        results = self.results[test_id]
        
        # Get robustness scores
        if 'robustness_scores' not in results:
            raise ValueError("No robustness scores found in results")
            
        scores = results['robustness_scores']
        
        # Create radar chart for robustness scores
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        
        # Categories to plot (excluding overall score)
        categories = [k for k in scores.keys() if k != 'overall_score']
        num_vars = len(categories)
        
        if num_vars == 0:
            raise ValueError("No category scores found in results")
            
        # Compute angle for each category
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Make the plot circular by repeating the first value
        angles += angles[:1]
        
        # Get scores for each category
        values = [scores[cat] for cat in categories]
        values += values[:1]  # Repeat the first value
        
        # Plot scores
        ax.plot(angles, values, 'o-', linewidth=2)
        
        # Fill the area
        ax.fill(angles, values, alpha=0.25)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        
        # Set radial ticks and limits
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)
        
        # Add overall score in the center
        plt.annotate(f"Overall: {scores['overall_score']:.2f}",
                    xy=(0.5, 0.5), xycoords='figure fraction',
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        plt.title(f"Robustness Summary - {test_id.replace('_', ' ').title()}")
        
        return fig
    
    def plot_feature_importance(self, test_id: Optional[str] = None, top_n: int = 10, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance based on robustness tests.
        
        Parameters:
        -----------
        test_id : str or None
            ID of test to plot (None for most recent)
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : Feature importance plot
        """
        # Get test results
        if test_id is None:
            if 'comprehensive_test' in self.results:
                test_id = 'comprehensive_test'
            elif self.results:
                test_id = list(self.results.keys())[-1]
            else:
                raise ValueError("No test results available")
                
        if test_id not in self.results:
            raise ValueError(f"Test results not found for ID: {test_id}")
            
        results = self.results[test_id]
        
        # Try to find feature importance in results
        feature_importance = None
        
        if 'feature_perturbation' in results:
            for test_type, test_result in results['feature_perturbation'].items():
                if test_type == 'all_features' and 'feature_importance' in test_result:
                    feature_importance = test_result['feature_importance']
                    break
        
        if feature_importance is None:
            # Run all_features test to get importance
            if self.verbose:
                print("Feature importance not found in results. Running all_features test...")
                
            all_features_result = self.feature_tests.test_all_features()
            feature_importance = all_features_result['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top N features
        if top_n and len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        y_pos = range(len(features))
        ax.barh(y_pos, importance_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Robustness Importance')
        ax.set_title('Feature Importance Based on Robustness')
        
        return fig