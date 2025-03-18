import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional

from deepbridge.visualization.robustness.base_viz import RobustnessBaseViz

class PerformanceViz(RobustnessBaseViz):
    """Visualizations related to model performance under perturbation."""
    
    @staticmethod
    def create_performance_chart(
        results: Dict, 
        metric_name: Optional[str] = None, 
        height: int = None, 
        width: int = None
    ):
        """
        Create an interactive performance comparison chart.
        
        Args:
            results: Results from RobustnessTest.evaluate_robustness
            metric_name: Name of the performance metric
            height: Height of the plot (now optional)
            width: Width of the plot (now optional)
            
        Returns:
            Plotly figure with the interactive chart
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        colors = ['#3A6EA5', '#FF6B6B', '#52D273', '#FFD166', '#6A67CE']
        
        # Add traces for each model
        for i, (model_name, model_results) in enumerate(results.items()):
            perturb_sizes = model_results['perturb_sizes']
            scores = model_results['mean_scores']
            
            # Calculate confidence intervals
            if len(model_results['all_scores']) > 0:
                stds = [np.std(scores_array) for scores_array in model_results['all_scores']]
                upper = [score + std for score, std in zip(scores, stds)]
                lower = [score - std for score, std in zip(scores, stds)]
                
                # Add confidence interval as a filled area
                fig.add_trace(
                    go.Scatter(
                        x=perturb_sizes + perturb_sizes[::-1],
                        y=upper + lower[::-1],
                        fill='toself',
                        fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name=f'{model_name} Confidence'
                    )
                )
            
            # Add line for main performance
            fig.add_trace(
                go.Scatter(
                    x=perturb_sizes,
                    y=scores,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,  # Reduced marker size
                        line=dict(color='white', width=1)  # Thinner marker border
                    ),
                    hovertemplate='Perturbation: %{x:.2f}<br>' +
                                f'Performance: %{{y:.4f}}<br>' +
                                f'Model: {model_name}'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Model Performance Under Perturbation" + 
                    (f" - {metric_name}" if metric_name else ""),
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)  # Reduced title size
            ),
            xaxis=dict(
                title=dict(
                    text="Perturbation Intensity",
                    font=dict(size=12)  # Reduced axis title size
                ),
                tickformat='.2f',
                gridcolor='rgba(230,230,230,0.5)'
            ),
            yaxis=dict(
                title=dict(
                    text=metric_name if metric_name else "Performance Metric",
                    font=dict(size=12)  # Reduced axis title size
                ),
                gridcolor='rgba(230,230,230,0.5)'
            ),
            plot_bgcolor='white',
            hovermode="closest",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=10)  # Smaller legend font
            ),
            margin=dict(l=40, r=40, t=80, b=60),  # Reduced margins
            autosize=True  # Enable autosize to fit container
        )
        
        # Add height and width only if specified
        if height:
            fig.update_layout(height=height)
        if width:
            fig.update_layout(width=width)
        
        # Add annotations for baseline and high perturbation with reduced size
        for i, (model_name, model_results) in enumerate(results.items()):
            scores = model_results['mean_scores']
            perturb_sizes = model_results['perturb_sizes']
            
            if len(scores) > 0:
                # Annotate extremes with smaller font and offset
                fig.add_annotation(
                    x=perturb_sizes[0],
                    y=scores[0],
                    text=f"Baseline: {scores[0]:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowwidth=1,
                    arrowcolor=colors[i % len(colors)],
                    ax=15,
                    ay=-20,
                    font=dict(size=9)
                )
                
                fig.add_annotation(
                    x=perturb_sizes[-1],
                    y=scores[-1],
                    text=f"Max Perturb: {scores[-1]:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowwidth=1,
                    arrowcolor=colors[i % len(colors)],
                    ax=-15,
                    ay=-20,
                    font=dict(size=9)
                )
        
        return fig

    @staticmethod
    def plot_boxplot_performance(
        results: Dict,
        model_name: Optional[str] = None,
        title: Optional[str] = None,
        metric_name: Optional[str] = None,
        height: int = 300,
        width: int = 900
    ):
        """
        Cria múltiplos boxplots organizados em subplots, semelhante ao estilo da imagem fornecida.
        
        Args:
            results: Dicionário com resultados do teste de robustez para cada modelo.
            title: Título personalizado para o gráfico.
            metric_name: Nome da métrica usada.
            height: Altura total da figura.
            width: Largura da figura.
        
        Returns:
            Figura Plotly com múltiplos boxplots organizados.
        """
        
        num_models = len(results)
        
        fig = make_subplots(
            rows=num_models,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Distribuição de Métricas para {model_name}" for model_name in results.keys()],
            vertical_spacing=0.05
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        
        for idx, (model_name, model_results) in enumerate(results.items(), start=1):
            perturb_sizes = model_results['perturb_sizes']
            all_scores = model_results['all_scores']
            
            for j, (size, scores) in enumerate(zip(perturb_sizes, all_scores)):
                fig.add_trace(go.Box(
                    y=scores,
                    x=[f"{size:.1f}"] * len(scores),
                    name=f"λ={size:.1f}",
                    marker_color=colors[j % len(colors)],
                    boxmean=True,
                    width=0.6,  # Aumentado para boxes maiores
                    showlegend=(idx == 1),
                    orientation='v'  # Fixar a orientação como vertical explicitamente
                ), row=idx, col=1)
        
        # Atualização geral do layout
        fig.update_layout(
            height=min(300*num_models, 600),  # Cap the height to prevent excessive stretching
            width=width,
            title_text=title or "Distribuição de Métricas por Modelo sob Perturbação",
            plot_bgcolor='white',
            boxmode='group',
            boxgroupgap=0.5, 
            boxgap=0.2,
            margin=dict(l=60, r=30, t=80, b=60)  # Add explicit margins
        )
        
        # Importante: configurar explicitamente cada subplot para preservar orientação
        for i in range(1, num_models+1):
            fig.update_xaxes(
                title_text="Tamanho da Perturbação (λ)" if i == num_models else None,
                type='category',
                row=i,
                col=1
            )
            
            fig.update_yaxes(
                title_text="Valor da Métrica" if i == 1 else None,
                gridcolor='rgba(230,230,230,0.5)',
                row=i,
                col=1,
                tickformat='.2f',  # Forçar formato com 2 casas decimais
                dtick=0.15,        # Definir intervalos de 0.05
                range=[0.5, 1.0]   # Fixar o intervalo de 0.5 a 1.0
            )
        
        if title:
            fig.update_layout(
                title=dict(text=title, x=0.5, y=0.98, font=dict(size=18))
            )
        
        return fig