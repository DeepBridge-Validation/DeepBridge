"""
Exemplo de uso das visualizações avançadas de robustez.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from deepbridge.validation.validators.robustness_validator import RobustnessValidator
from deepbridge.validation.visualization.robustness_plots import RobustnessPlotter, RobustnessPlotConfig

def main():
    # Gera dados de exemplo
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    # Converte para DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    
    # Cria e treina modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Configura validador de robustez
    validator = RobustnessValidator(
        model=model,
        perturbation_types=['noise', 'zero', 'missing'],
        perturbation_levels=[0.1, 0.2, 0.5],
        random_state=42,
        verbose=True
    )
    
    # Executa validação
    results = validator.validate(X, y)
    
    # Configurar visualizações
    config = RobustnessPlotConfig(
        template='plotly_white',
        width=1200,
        height=800
    )
    
    # Criar plotter
    plotter = RobustnessPlotter(results, config)
    
    # Gerar visualizações
    fig_importance = plotter.create_feature_importance_plot()
    fig_heatmap = plotter.create_robustness_heatmap()
    
    # Cria diferentes visualizações
    # 1. Gráfico de importância de features
    fig_importance.write_html('feature_importance.html')
    
    # 2. Gráfico de impacto das perturbações (para cada tipo)
    for p_type in ['noise', 'zero', 'missing']:
        fig_impact = plotter.create_perturbation_impact_plot(p_type)
        fig_impact.write_html(f'perturbation_impact_{p_type}.html')
    
    # 3. Mapa de calor de robustez
    fig_heatmap.write_html('robustness_heatmap.html')
    
    # 4. Animação de perturbações
    for p_type in ['noise', 'zero', 'missing']:
        fig_animation = plotter.create_perturbation_animation(p_type)
        fig_animation.write_html(f'perturbation_animation_{p_type}.html')

if __name__ == '__main__':
    main() 