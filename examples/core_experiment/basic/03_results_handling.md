# Manipulação de Resultados

Este exemplo demonstra como trabalhar com os resultados de um experimento, incluindo acesso, processamento e visualização dos resultados.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.results import wrap_results

# 1. Configurar um experimento básico
# ----------------------------------
# Carregar e preparar dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Criar DBDataset
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=model,
    dataset_name="Breast Cancer"
)

# Criar experimento
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty"],  # Especificar os testes que serão executados
    random_state=42
)

# 2. Executar testes e obter resultados
# -----------------------------------
# Executar os testes com configuração "quick" (mais rápida)
results = experiment.run_tests(config_name="quick")

print(f"Testes executados: {list(results.results.keys())}")

# 3. Acessar resultados iniciais
# ----------------------------
initial_results = results.results['initial_results']

print("\nMétricas do modelo principal:")
if 'models' in initial_results and 'primary_model' in initial_results['models']:
    metrics = initial_results['models']['primary_model']['metrics']
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# 4. Acessar resultados de testes específicos
# -----------------------------------------
# Testes de robustez
if 'robustness' in results.results:
    robustness_results = results.results['robustness']
    print("\nResultados do teste de robustez:")
    
    # Pontuação geral de robustez
    if 'overall_robustness_score' in robustness_results:
        print(f"  Pontuação geral de robustez: {robustness_results['overall_robustness_score']:.4f}")
    
    # Robustez por feature
    if 'feature_robustness' in robustness_results:
        feature_robustness = robustness_results['feature_robustness']
        print("\n  Robustez por feature (top 5):")
        for feature, score in sorted(feature_robustness.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {feature}: {score:.4f}")

# Testes de incerteza
if 'uncertainty' in results.results:
    uncertainty_results = results.results['uncertainty']
    print("\nResultados do teste de incerteza:")
    
    # Métricas principais de incerteza
    for metric in ['calibration_error', 'sharpness', 'coverage_error']:
        if metric in uncertainty_results:
            print(f"  {metric}: {uncertainty_results[metric]:.4f}")

# 5. Métodos alternativos para acessar resultados específicos
# ---------------------------------------------------------
# Usando os métodos de acesso direto do experimento
robustness_via_method = experiment.get_robustness_results()
uncertainty_via_method = experiment.get_uncertainty_results()

print("\nAcessando resultados via métodos específicos:")
print(f"  Robustez via método: {len(robustness_via_method)} itens")
print(f"  Incerteza via método: {len(uncertainty_via_method)} itens")

# 6. Visualizar resultados
# ----------------------
# Criar pasta para salvar visualizações
output_dir = "experiment_results"
os.makedirs(output_dir, exist_ok=True)

# Visualizar importância das features
if 'model_feature_importance' in results.results:
    feature_importance = results.results['model_feature_importance']
    
    # Ordenar por importância
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_features[:10]]  # Top 10 features
    importances = [x[1] for x in sorted_features[:10]]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.title('Top 10 Features por Importância')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    print(f"\nVisualizações salvas em: {output_dir}/feature_importance.png")

# Visualizar robustez por feature
if 'robustness' in results.results and 'feature_robustness' in results.results['robustness']:
    feature_robustness = results.results['robustness']['feature_robustness']
    
    # Ordenar por robustez
    sorted_robustness = sorted(feature_robustness.items(), key=lambda x: x[1])
    features = [x[0] for x in sorted_robustness[:10]]  # 10 features menos robustas
    rob_scores = [x[1] for x in sorted_robustness[:10]]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, rob_scores)
    plt.xlabel('Pontuação de Robustez')
    plt.ylabel('Feature')
    plt.title('10 Features Menos Robustas')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_robustness.png'))
    plt.close()
    
    print(f"Visualizações salvas em: {output_dir}/feature_robustness.png")

# 7. Gerar um relatório HTML
# ------------------------
# Gerar relatório para teste de robustez
robustness_report = os.path.join(output_dir, 'robustness_report.html')
report_path = experiment.save_html(
    test_type="robustness",
    file_path=robustness_report,
    model_name="Modelo Random Forest"
)

print(f"\nRelatório HTML gerado em: {report_path}")

# 8. Serializar e carregar resultados
# ---------------------------------
# Salvar resultados em formato JSON
import json

# Serializar apenas os resultados de alto nível (alguns objetos complexos não são serializáveis)
json_results = {
    'experiment_type': experiment.experiment_type,
    'tests_executed': list(results.results.keys()),
    'model_metrics': initial_results['models']['primary_model']['metrics'],
    'robustness_score': results.results.get('robustness', {}).get('overall_robustness_score', None),
    'uncertainty_metrics': {
        k: v for k, v in results.results.get('uncertainty', {}).items() 
        if isinstance(v, (int, float, str, list, dict))
    }
}

# Salvar em arquivo JSON
json_path = os.path.join(output_dir, 'experiment_results.json')
with open(json_path, 'w') as f:
    json.dump(json_results, f, indent=2)

print(f"Resultados serializados para JSON em: {json_path}")

# 9. Combinar resultados de múltiplos experimentos
# ---------------------------------------------
# Criar um segundo experimento com um modelo diferente
model2 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model2.fit(X_train, y_train)

dataset2 = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=model2,
    dataset_name="Breast Cancer (Modelo 2)"
)

experiment2 = Experiment(
    dataset=dataset2,
    experiment_type="binary_classification",
    tests=["robustness"],
    random_state=42
)

# Executar testes no segundo experimento
results2 = experiment2.run_tests(config_name="quick")

# Combinar resultados para comparação
combined_results = {
    'model1': {
        'accuracy': initial_results['models']['primary_model']['metrics'].get('accuracy'),
        'robustness': results.results.get('robustness', {}).get('overall_robustness_score')
    },
    'model2': {
        'accuracy': results2.results['initial_results']['models']['primary_model']['metrics'].get('accuracy'),
        'robustness': results2.results.get('robustness', {}).get('overall_robustness_score')
    }
}

# Criar DataFrame para visualização
comparison_df = pd.DataFrame(combined_results).T
print("\nComparação dos modelos:")
print(comparison_df)

# Plotar comparação
plt.figure(figsize=(10, 6))
comparison_df.plot(kind='bar', ax=plt.gca())
plt.title('Comparação entre Modelos')
plt.ylabel('Pontuação')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
plt.close()

print(f"Comparação visual salva em: {output_dir}/model_comparison.png")

# 10. Uso avançado: Wrap dos resultados manualmente
# -----------------------------------------------
# Criar um dicionário personalizado de resultados
custom_results = {
    'experiment_type': 'binary_classification',
    'config': {'name': 'custom', 'tests': ['robustness']},
    'custom_data': {
        'model_accuracy': 0.95,
        'feature_importance': {'feature_1': 0.5, 'feature_2': 0.3},
        'custom_metric': 'valor personalizado'
    }
}

# Usar a função wrap_results para adicionar funcionalidades adicionais
wrapped_results = wrap_results(custom_results)

# Agora o objeto wrapped_results terá métodos como save_html
print("\nResultados customizados e enriquecidos com métodos adicionais:")
print(f"  Tipo do objeto: {type(wrapped_results)}")
print(f"  Métodos disponíveis: save_html, ...")
```

## Estrutura dos Resultados

Os resultados de um experimento no DeepBridge estão estruturados hierarquicamente:

```
results
│
├── experiment_type          # Tipo do experimento (binary_classification, regression, etc.)
├── config                   # Configuração usada no experimento
│
├── initial_results          # Resultados básicos (sempre presente)
│   ├── models               # Resultados de todos os modelos
│   │   ├── primary_model    # Modelo principal
│   │   │   ├── metrics      # Métricas do modelo
│   │   │   └── feature_importance  # Importância das features (se disponível)
│   │   │
│   │   └── [outros modelos] # Modelos alternativos
│
├── robustness               # Resultados do teste de robustez (se executado)
│   ├── overall_robustness_score  # Pontuação geral de robustez
│   ├── feature_robustness   # Robustez por feature
│   └── ...
│
├── uncertainty              # Resultados do teste de incerteza (se executado)
│   ├── calibration_error    # Erro de calibração
│   ├── coverage_error       # Erro de cobertura
│   └── ...
│
├── resilience               # Resultados do teste de resiliência (se executado)
└── hyperparameters          # Resultados de importância de hiperparâmetros (se executado)
```

## Métodos de Acesso aos Resultados

A classe `Experiment` oferece vários métodos para acessar os resultados:

### Acesso Direto
- `experiment.initial_results`: Resultados iniciais do experimento
- `experiment.test_results`: Todos os resultados de testes

### Métodos Específicos
- `experiment.get_robustness_results()`: Resultados do teste de robustez
- `experiment.get_uncertainty_results()`: Resultados do teste de incerteza
- `experiment.get_resilience_results()`: Resultados do teste de resiliência
- `experiment.get_hyperparameter_results()`: Resultados do teste de hiperparâmetros
- `experiment.get_feature_importance()`: Importância das features para o modelo principal

## Geração de Relatórios

O DeepBridge pode gerar relatórios HTML interativos para visualizar os resultados dos testes:

- `experiment.save_html(test_type, file_path, model_name)`: Gera e salva um relatório HTML
  - **test_type**: Tipo de teste ('robustness', 'uncertainty', etc.)
  - **file_path**: Caminho onde o relatório será salvo
  - **model_name**: Nome do modelo para exibição no relatório

## Pontos-Chave

- Os resultados são estruturados hierarquicamente com resultados iniciais e resultados de testes específicos
- Você pode acessar diretamente os resultados ou usar métodos específicos
- Os resultados podem ser visualizados através de gráficos ou relatórios HTML interativos
- É possível serializar resultados para armazenamento e compartilhamento
- A função `wrap_results()` enriquece um dicionário de resultados com métodos adicionais como `save_html`
- A comparação de resultados de múltiplos experimentos permite avaliar diferentes modelos ou configurações