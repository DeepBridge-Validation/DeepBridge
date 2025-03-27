"""
Visualizações avançadas para testes de robustez usando Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class RobustnessPlotConfig:
    """Configuração para visualizações de robustez."""
    template: str = "plotly_white"
    width: int = 1200
    height: int = 800
    show_legend: bool = True
    animation_duration: int = 500
    color_sequence: List[str] = None

class RobustnessPlotter:
    """Classe para criar visualizações avançadas de testes de robustez."""
    
    def __init__(self, results: Dict[str, Any], config: Optional[RobustnessPlotConfig] = None):
        """
        Inicializa o plotter de robustez.
        
        Parameters:
        -----------
        results : dict
            Resultados dos testes de robustez
        config : RobustnessPlotConfig, optional
            Configurações de visualização
        """
        self.results = results
        self.config = config or RobustnessPlotConfig()
        
        # Define sequência de cores padrão se não fornecida
        if self.config.color_sequence is None:
            self.config.color_sequence = px.colors.qualitative.Set3
            
    def create_feature_importance_plot(self) -> go.Figure:
        """
        Cria um gráfico interativo de importância de features.
        
        Returns:
        --------
        go.Figure : Figura Plotly
        """
        # Extrai dados de importância
        feature_importance = self.results.get('feature_importance', {})
        
        # Cria DataFrame para plotagem
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True)
        
        # Cria figura
        fig = go.Figure()
        
        # Adiciona barras horizontais
        fig.add_trace(
            go.Bar(
                y=df['Feature'],
                x=df['Importance'],
                orientation='h',
                marker_color=self.config.color_sequence[0],
                name='Importância'
            )
        )
        
        # Atualiza layout
        fig.update_layout(
            title='Importância das Features para Robustez',
            xaxis_title='Score de Importância',
            yaxis_title='Features',
            template=self.config.template,
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_perturbation_impact_plot(self, perturbation_type: str) -> go.Figure:
        """
        Cria um gráfico interativo do impacto das perturbações.
        
        Parameters:
        -----------
        perturbation_type : str
            Tipo de perturbação a ser visualizada
            
        Returns:
        --------
        go.Figure : Figura Plotly
        """
        # Extrai dados de perturbação
        perturbation_results = self.results.get('perturbations', {}).get(perturbation_type, {})
        feature_results = perturbation_results.get('feature_results', {})
        
        # Cria subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Impacto nas Métricas', 'Mudança Relativa'),
            vertical_spacing=0.2
        )
        
        # Adiciona traços para cada feature
        for idx, (feature, results) in enumerate(feature_results.items()):
            # Extrai dados
            levels = [r['level'] for r in results]
            performance = [r['performance'] for r in results]
            relative_change = [r['relative_change'] for r in results]
            
            # Adiciona traços de performance
            for metric in performance[0].keys():
                metric_values = [p[metric] for p in performance]
                fig.add_trace(
                    go.Scatter(
                        x=levels,
                        y=metric_values,
                        name=f'{feature} - {metric}',
                        mode='lines+markers',
                        line=dict(color=self.config.color_sequence[idx % len(self.config.color_sequence)]),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Adiciona traços de mudança relativa
            for metric in relative_change[0].keys():
                change_values = [rc[metric] for rc in relative_change]
                fig.add_trace(
                    go.Scatter(
                        x=levels,
                        y=change_values,
                        name=f'{feature} - {metric}',
                        mode='lines+markers',
                        line=dict(color=self.config.color_sequence[idx % len(self.config.color_sequence)]),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # Atualiza layout
        fig.update_layout(
            title=f'Impacto das Perturbações do Tipo: {perturbation_type}',
            template=self.config.template,
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend
        )
        
        # Atualiza eixos
        fig.update_xaxes(title_text='Nível de Perturbação', row=1, col=1)
        fig.update_xaxes(title_text='Nível de Perturbação', row=2, col=1)
        fig.update_yaxes(title_text='Valor da Métrica', row=1, col=1)
        fig.update_yaxes(title_text='Mudança Relativa', row=2, col=1)
        
        return fig
    
    def create_robustness_heatmap(self) -> go.Figure:
        """
        Cria um mapa de calor interativo da robustez por feature e tipo de perturbação.
        
        Returns:
        --------
        go.Figure : Figura Plotly
        """
        # Extrai dados
        perturbations = self.results.get('perturbations', {})
        
        # Prepara dados para o mapa de calor
        features = set()
        perturbation_types = set()
        robustness_scores = {}
        
        for p_type, results in perturbations.items():
            perturbation_types.add(p_type)
            for feature, feature_results in results.get('feature_results', {}).items():
                features.add(feature)
                # Calcula score de robustez para feature
                scores = [r['relative_change']['accuracy'] for r in feature_results]
                robustness_scores[(feature, p_type)] = np.mean(scores)
        
        # Cria matriz de dados
        data = []
        for feature in features:
            row = []
            for p_type in perturbation_types:
                row.append(robustness_scores.get((feature, p_type), 0))
            data.append(row)
        
        # Cria figura
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=list(perturbation_types),
            y=list(features),
            colorscale='RdYlGn_r',  # Vermelho para menos robusto, verde para mais robusto
            colorbar=dict(title='Score de Robustez')
        ))
        
        # Atualiza layout
        fig.update_layout(
            title='Mapa de Calor de Robustez por Feature e Tipo de Perturbação',
            xaxis_title='Tipo de Perturbação',
            yaxis_title='Features',
            template=self.config.template,
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_perturbation_animation(self, perturbation_type: str) -> go.Figure:
        """
        Cria uma animação interativa do impacto das perturbações.
        
        Parameters:
        -----------
        perturbation_type : str
            Tipo de perturbação a ser visualizada
            
        Returns:
        --------
        go.Figure : Figura Plotly
        """
        # Extrai dados
        perturbation_results = self.results.get('perturbations', {}).get(perturbation_type, {})
        feature_results = perturbation_results.get('feature_results', {})
        
        # Cria figura
        fig = go.Figure()
        
        # Adiciona frames para cada nível de perturbação
        frames = []
        levels = sorted(list(set(
            level for results in feature_results.values()
            for level in [r['level'] for r in results]
        )))
        
        for level in levels:
            frame_data = []
            for feature, results in feature_results.items():
                result = next((r for r in results if r['level'] == level), None)
                if result:
                    frame_data.append(
                        go.Bar(
                            name=feature,
                            x=[feature],
                            y=[result['performance']['accuracy']],
                            text=[f"{result['performance']['accuracy']:.3f}"],
                            textposition='auto',
                        )
                    )
            
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=f'Level {level:.2f}',
                    layout=go.Layout(
                        title=f'Impacto da Perturbação {perturbation_type} (Nível: {level:.2f})'
                    )
                )
            )
        
        # Adiciona dados iniciais
        fig.add_trace(frames[0].data[0])
        
        # Atualiza layout
        fig.update_layout(
            template=self.config.template,
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend,
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {'frame': {'duration': self.config.animation_duration, 'redraw': True},
                               'fromcurrent': True}]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], {'frame': {'duration': 0, 'redraw': True},
                                  'mode': 'immediate',
                                  'transition': {'duration': 0}}]
                )]
            )]
        )
        
        # Adiciona frames
        fig.frames = frames
        
        return fig 