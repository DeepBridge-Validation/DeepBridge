# Testes de Hiperparâmetros

Este exemplo demonstra como realizar testes de hiperparâmetros usando o módulo `core.experiment`. Os testes de hiperparâmetros avaliam a importância e o impacto de diferentes hiperparâmetros no desempenho do modelo.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.managers.hyperparameter_manager import HyperparameterManager

# 1. Preparar dados
# ---------------
# Carregar dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Treinar modelo com hiperparâmetros específicos
# -----------------------------------------------
# Configuração de hiperparâmetros para analisar
hyperparams = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42
}

# Criar e treinar o modelo
model = RandomForestClassifier(**hyperparams)
model.fit(X_train, y_train)
print(f"Modelo treinado com hiperparâmetros personalizados")

# 3. Configurar e executar teste de hiperparâmetros via Experiment
# -------------------------------------------------------------
# Criar DataFrames de treino e teste
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Criar DBDataset
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=model,
    dataset_name="Random Forest"
)

# Criar experimento focado em testes de hiperparâmetros
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["hyperparameters"],  # Apenas teste de hiperparâmetros
    random_state=42
)

# Executar teste de hiperparâmetros com configuração "quick"
print("\nExecutando teste de hiperparâmetros (pode levar algum tempo)...")
results = experiment.run_tests(config_name="quick")

# 4. Analisar resultados dos testes de hiperparâmetros
# -------------------------------------------------
hyperparameter_results = results.results.get('hyperparameters', {})

print("\nResultados do teste de hiperparâmetros:")
print(f"Teste concluído: {'hyperparameters' in results.results}")

# 5. Visualizar importância dos hiperparâmetros
# -------------------------------------------
if 'hyperparameter_importance' in hyperparameter_results:
    importance = hyperparameter_results['hyperparameter_importance']
    
    # Converter para DataFrame para fácil visualização
    importance_df = pd.DataFrame({
        'Hiperparâmetro': list(importance.keys()),
        'Importância': list(importance.values())
    }).sort_values('Importância', ascending=False)
    
    print("\nImportância dos hiperparâmetros:")
    print(importance_df)
    
    # Visualizar importância
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importância', y='Hiperparâmetro', data=importance_df)
    plt.title('Importância dos Hiperparâmetros')
    plt.tight_layout()
    plt.show()

# 6. Visualizar efeitos parciais dos hiperparâmetros
# ------------------------------------------------
if 'partial_dependence' in hyperparameter_results:
    partial_dependence = hyperparameter_results['partial_dependence']
    
    # Selecionar hiperparâmetros mais importantes para visualização
    top_hyperparams = importance_df['Hiperparâmetro'].tolist()[:3]
    
    # Criar visualizações para os hiperparâmetros mais importantes
    plt.figure(figsize=(15, 5))
    
    for i, hyperparam in enumerate(top_hyperparams):
        if hyperparam in partial_dependence:
            pd_data = partial_dependence[hyperparam]
            
            plt.subplot(1, 3, i+1)
            
            # Verificar se os dados estão no formato esperado
            if isinstance(pd_data, dict) and 'values' in pd_data and 'effects' in pd_data:
                values = pd_data['values']
                effects = pd_data['effects']
                
                # Converter para o tipo correto para visualização
                if isinstance(values[0], str):
                    # Para hiperparâmetros categóricos
                    plt.bar(values, effects)
                else:
                    # Para hiperparâmetros numéricos
                    plt.plot(values, effects, marker='o')
                    
                plt.title(f'Efeito de {hyperparam}')
                plt.xlabel(hyperparam)
                plt.ylabel('Impacto na Performance')
                
    plt.tight_layout()
    plt.show()

# 7. Analisar resultados da otimização
# ----------------------------------
if 'optimization_results' in hyperparameter_results:
    optimization = hyperparameter_results['optimization_results']
    
    # Extrair melhores hiperparâmetros
    if 'best_params' in optimization:
        best_params = optimization['best_params']
        best_score = optimization.get('best_score')
        
        print("\nMelhores hiperparâmetros encontrados:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        if best_score is not None:
            print(f"Melhor score: {best_score:.4f}")
    
    # Visualizar histórico de otimização se disponível
    if 'history' in optimization:
        history = optimization['history']
        
        if isinstance(history, list) and len(history) > 0:
            # Converter histórico para DataFrame
            history_df = pd.DataFrame(history)
            
            if 'trial_number' in history_df and 'score' in history_df:
                plt.figure(figsize=(10, 6))
                plt.plot(history_df['trial_number'], history_df['score'], marker='o')
                plt.title('Progresso da Otimização de Hiperparâmetros')
                plt.xlabel('Número do Trial')
                plt.ylabel('Score')
                plt.grid(True)
                plt.show()

# 8. Uso direto da classe HyperparameterManager
# -------------------------------------------
# Instanciar o gerenciador de hiperparâmetros diretamente
hyperparam_mgr = HyperparameterManager(
    dataset=dataset,
    verbose=True
)

# Especificar espaço de hiperparâmetros personalizado
param_space = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Executar teste personalizado
custom_results = hyperparam_mgr.run_hyperparameter_test(
    param_space=param_space,
    n_trials=20,  # Número reduzido para este exemplo
    scoring='roc_auc',  # Métrica para otimização
    random_state=42
)

print("\nResultados personalizados do teste de hiperparâmetros:")

# Mostrar melhores hiperparâmetros encontrados
if 'optimization_results' in custom_results and 'best_params' in custom_results['optimization_results']:
    best_params = custom_results['optimization_results']['best_params']
    best_score = custom_results['optimization_results'].get('best_score')
    
    print("Melhores hiperparâmetros encontrados (teste personalizado):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    if best_score is not None:
        print(f"Melhor score: {best_score:.4f}")

# 9. Treinar modelo com os melhores hiperparâmetros
# -----------------------------------------------
# Usar os melhores hiperparâmetros do teste personalizado
if 'optimization_results' in custom_results and 'best_params' in custom_results['optimization_results']:
    best_params = custom_results['optimization_results']['best_params']
    
    # Criar e treinar o modelo com os melhores hiperparâmetros
    optimized_model = RandomForestClassifier(**best_params)
    optimized_model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    accuracy = optimized_model.score(X_test, y_test)
    
    print(f"\nDesempenho do modelo otimizado no conjunto de teste: {accuracy:.4f}")
    
    # Comparar com o modelo original
    original_accuracy = model.score(X_test, y_test)
    print(f"Desempenho do modelo original no conjunto de teste: {original_accuracy:.4f}")
    print(f"Melhoria: {(accuracy - original_accuracy) * 100:.2f}%")

# 10. Visualização avançada: Interação entre hiperparâmetros
# -------------------------------------------------------
# Analisar a interação entre pares de hiperparâmetros mais importantes
if 'optimization_results' in custom_results and 'trials' in custom_results['optimization_results']:
    trials = custom_results['optimization_results']['trials']
    
    if isinstance(trials, list) and len(trials) > 0:
        # Converter trials para DataFrame
        trials_df = pd.DataFrame(trials)
        
        # Selecionar os dois hiperparâmetros mais importantes
        top_params = list(importance_df['Hiperparâmetro'].values)[:2] if 'importance_df' in locals() else ['n_estimators', 'max_depth']
        
        # Verificar se ambos estão presentes
        if all(param in trials_df.columns for param in top_params) and 'score' in trials_df.columns:
            # Criar um gráfico de dispersão para visualizar a interação
            plt.figure(figsize=(10, 8))
            
            # Converter valores não numéricos se necessário
            for param in top_params:
                if trials_df[param].dtype == 'object':
                    # Criar um mapeamento para valores numéricos
                    unique_values = trials_df[param].unique()
                    value_map = {val: i for i, val in enumerate(unique_values)}
                    trials_df[f"{param}_numeric"] = trials_df[param].map(value_map)
                    top_params[top_params.index(param)] = f"{param}_numeric"
            
            # Criar scatter plot
            scatter = plt.scatter(
                trials_df[top_params[0]], 
                trials_df[top_params[1]], 
                c=trials_df['score'], 
                cmap='viridis', 
                s=100, 
                alpha=0.7
            )
            
            plt.colorbar(scatter, label='Score')
            plt.xlabel(top_params[0].replace('_numeric', ''))
            plt.ylabel(top_params[1].replace('_numeric', ''))
            plt.title('Interação entre Hiperparâmetros Importantes')
            plt.grid(True)
            plt.show()

# 11. Gerar e salvar relatório HTML
# ------------------------------
# Criar diretório para relatórios
import os
output_dir = "hyperparameter_reports"
os.makedirs(output_dir, exist_ok=True)

# Gerar relatório
report_path = os.path.join(output_dir, "hyperparameter_report.html")
experiment.save_html(
    test_type="hyperparameters",
    file_path=report_path,
    model_name="Random Forest"
)

print(f"\nRelatório de hiperparâmetros salvo em: {report_path}")
```

## Como Funcionam os Testes de Hiperparâmetros

Os testes de hiperparâmetros no DeepBridge avaliam a importância e o impacto de diferentes hiperparâmetros no desempenho do modelo. O processo inclui:

1. **Otimização de Hiperparâmetros**: Busca sistemática pelos melhores hiperparâmetros
2. **Análise de Importância**: Avaliação da contribuição relativa de cada hiperparâmetro
3. **Análise de Dependência Parcial**: Visualização do efeito individual de cada hiperparâmetro

### Componentes do Teste

- **Otimização de Hiperparâmetros**: Utiliza otimização bayesiana (Optuna) para encontrar configurações ótimas
- **Análise de Importância**: Calcula quanto cada hiperparâmetro contribui para a variação no desempenho
- **Dependência Parcial**: Mostra como o desempenho varia em função de um hiperparâmetro específico

### Configurações do Teste

- **param_space**: Espaço de busca para cada hiperparâmetro
- **n_trials**: Número de combinações de hiperparâmetros a testar
- **scoring**: Métrica a ser otimizada (ex: 'accuracy', 'roc_auc', 'f1')
- **config_name**: "quick", "medium", ou "full" (predefinições que controlam a profundidade do teste)

## Interpretando os Resultados

### Importância dos Hiperparâmetros

- **Alta Importância**: Hiperparâmetros com grande impacto no desempenho do modelo
- **Baixa Importância**: Hiperparâmetros que podem ser deixados com valores padrão

A importância é calculada com base na variação do desempenho quando um hiperparâmetro é modificado, controlando para os outros hiperparâmetros.

### Dependência Parcial

Mostra como o desempenho do modelo varia em função de um único hiperparâmetro. Isto ajuda a:

- Identificar valores ótimos para cada hiperparâmetro
- Entender a sensibilidade do modelo a cada hiperparâmetro
- Visualizar relações não-lineares entre hiperparâmetros e desempenho

### Resultados da Otimização

- **Melhores Hiperparâmetros**: Combinação que produziu o melhor desempenho
- **Melhor Score**: Valor da métrica de desempenho para os melhores hiperparâmetros
- **Histórico de Trials**: Evolução do desempenho durante a otimização

## Aplicações Práticas

### Entendimento do Modelo

- Identificar quais hiperparâmetros são mais críticos para o modelo
- Entender como cada hiperparâmetro afeta o desempenho
- Descobrir interações importantes entre hiperparâmetros

### Otimização do Modelo

- Encontrar configurações que maximizam o desempenho
- Reduzir o tempo de ajuste fino focando nos hiperparâmetros mais importantes
- Obter insights para melhorar o modelo

### Trade-offs

- Balancear desempenho e complexidade do modelo
- Identificar pontos de diminuição de retorno em hiperparâmetros como `n_estimators`
- Encontrar configurações que oferecem bom equilíbrio entre diferentes métricas

## Valores de Configuração (config_name)

- **quick**: Busca rápida com poucas iterações (~20 trials)
- **medium**: Busca mais abrangente (~50 trials)
- **full**: Busca extensa para otimização completa (~100 trials)

## Pontos-Chave

- Os testes de hiperparâmetros ajudam a entender e otimizar o comportamento do modelo
- A importância dos hiperparâmetros revela quais configurações merecem mais atenção
- A dependência parcial mostra como cada hiperparâmetro afeta o desempenho do modelo
- A otimização bayesiana permite encontrar boas configurações eficientemente
- O relatório HTML fornece visualizações interativas dos resultados
- O conhecimento obtido pode ser usado para melhorar modelos futuros