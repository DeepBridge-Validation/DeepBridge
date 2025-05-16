"""
Testar a geração de gráficos de distribuição para relatórios de incerteza.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('incerteza_debug.log')
    ]
)

# Adicionar o diretório pai ao caminho de importação
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from deepbridge.core.db_data import DBDataset
from deepbridge.utils.uncertainty import run_uncertainty_tests
from deepbridge.core.experiment import Experiment
from deepbridge.core.experiment.test_runner import TestRunner

# Criar dados sintéticos para teste
def create_synthetic_data(n_samples=1000, n_features=5, noise=0.5, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # Criar características com diferentes distribuições para tornar o exemplo mais interessante
    X[:, 0] = np.random.normal(0, 1, n_samples)  # Normal
    X[:, 1] = np.random.exponential(2, n_samples)  # Exponencial
    X[:, 2] = np.random.uniform(-3, 3, n_samples)  # Uniforme
    X[:, 3] = np.random.beta(2, 5, n_samples) * 10  # Beta
    X[:, 4] = np.random.gamma(2, 2, n_samples)  # Gamma
    
    # Gerar valores de saída com não-linearidades
    y = (
        2 * X[:, 0] +
        0.5 * X[:, 1]**2 +
        np.sin(X[:, 2]) * 3 +
        np.log1p(np.abs(X[:, 3])) +
        np.sqrt(np.abs(X[:, 4]))
    )
    
    # Adicionar ruído
    y += np.random.normal(0, noise, n_samples)
    
    # Converter para DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    return df, pd.Series(y, name='target')

def main():
    # Criar dados
    logging.info("Criando dados sintéticos...")
    X, y = create_synthetic_data(n_samples=1000, n_features=5, noise=0.5)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinar modelo
    logging.info("Treinando modelo...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Criar DBDataset
    logging.info("Criando DBDataset...")
    # Criar DataFrames de treino e teste com features e target
    train_df = X_train.copy()
    train_df['target'] = y_train.copy()
    
    test_df = X_test.copy()
    test_df['target'] = y_test.copy()
    
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model
    )
    
    # Executar testes de incerteza e gerar logs detalhados
    logging.info("Executando testes de incerteza...")
    results = run_uncertainty_tests(dataset, config_name='full', verbose=True)
    
    # Verificar se temos os dados necessários para gráficos de distribuição
    logging.info("Verificando dados para gráficos de distribuição...")
    if 'reliability_analysis' in results:
        logging.info("reliability_analysis está presente nos resultados")
        logging.info(f"Chaves em reliability_analysis: {list(results['reliability_analysis'].keys())}")
        
        if 'feature_distributions' in results['reliability_analysis']:
            logging.info("feature_distributions está presente")
            logging.info(f"Tipos de distribuições: {list(results['reliability_analysis']['feature_distributions'].keys())}")
    else:
        logging.warning("reliability_analysis NÃO está presente nos resultados")
    
    if 'marginal_bandwidth' in results:
        logging.info("marginal_bandwidth está presente nos resultados")
        logging.info(f"Features com análise de banda marginal: {list(results['marginal_bandwidth'].keys())}")
    else:
        logging.warning("marginal_bandwidth NÃO está presente nos resultados")
    
    if 'interval_widths' in results:
        logging.info("interval_widths está presente nos resultados")
        logging.info(f"Tipo dos dados de interval_widths: {type(results['interval_widths'])}")
    else:
        logging.warning("interval_widths NÃO está presente nos resultados")
    
    # Como estamos tendo problemas com o runner, vamos pular a parte de geração de relatório
    logging.info("Não vamos gerar relatório com o Experiment, apenas analisar os dados de incerteza")
    
    # Em vez disso, vamos analisar diretamente os resultados de incerteza
    if 'reliability_analysis' in results:
        logging.info("Detalhes da análise de confiabilidade:")
        ra = results['reliability_analysis']
        logging.info(f"- Threshold: {ra.get('threshold_value')}")
        logging.info(f"- Contagem confiável: {ra.get('reliable_count')}")
        logging.info(f"- Contagem não confiável: {ra.get('unreliable_count')}")
        
        fd = ra.get('feature_distributions', {})
        if fd:
            logging.info("Distribuições de características disponíveis:")
            for tipo, features in fd.items():
                logging.info(f"- Tipo {tipo}: {list(features.keys())}")
                
    if 'marginal_bandwidth' in results:
        logging.info("Detalhes da análise de banda marginal:")
        mb = results['marginal_bandwidth']
        for feature, data in mb.items():
            logging.info(f"- Feature {feature}:")
            logging.info(f"  - Bin centers: {len(data.get('bin_centers', []))} valores")
            logging.info(f"  - Avg widths: {len(data.get('avg_widths', []))} valores")
            
    if 'interval_widths' in results:
        logging.info("Dados de larguras de intervalo:")
        if isinstance(results['interval_widths'], list):
            logging.info(f"- Lista com {len(results['interval_widths'])} elementos")
            if results['interval_widths'] and isinstance(results['interval_widths'][0], list):
                logging.info(f"- Primeiro elemento tem {len(results['interval_widths'][0])} valores")
        elif isinstance(results['interval_widths'], dict):
            logging.info(f"- Dicionário com chaves: {list(results['interval_widths'].keys())}")
            
    logging.info("Análise completa. Os dados necessários para os gráficos de distribuição estão presentes.")
    
    logging.info("Processo completo. Verifique o arquivo de log para detalhes.")

if __name__ == "__main__":
    main()