# Testes de Robustez

Este exemplo demonstra como realizar testes de robustez de modelos usando o módulo `core.experiment`. Os testes de robustez avaliam quão bem um modelo mantém seu desempenho quando as features de entrada são perturbadas.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.managers.robustness_manager import RobustnessManager
from deepbridge.validation.wrappers.robustness import data_perturber

# 1. Preparar dados e modelo
# --------------------------
# Carregar dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelos para comparação
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Modelo {name} treinado")

# 2. Configurar e executar teste de robustez via Experiment
# --------------------------------------------------------
# Criar DBDataset para o modelo Random Forest
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

rf_dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=models['random_forest'],
    dataset_name="Random Forest"
)

# Criar experimento focado em testes de robustez
experiment = Experiment(
    dataset=rf_dataset,
    experiment_type="binary_classification",
    tests=["robustness"],  # Apenas teste de robustez
    # Opcionalmente, podemos focar em um subconjunto de features
    feature_subset=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],
    random_state=42
)

# Executar teste de robustez com configuração "quick"
# Configurações disponíveis: "quick", "medium", "full"
results = experiment.run_tests(config_name="quick")

# Acessar resultados do teste de robustez
robustness_results = results.results['robustness']

print("\nResultados do teste de robustez:")
print(f"Pontuação geral de robustez: {robustness_results['overall_robustness_score']:.4f}")

# 3. Examinar robustez por feature
# -------------------------------
if 'feature_robustness' in robustness_results:
    feature_robustness = robustness_results['feature_robustness']
    
    # Ordenar features por robustez (do menos ao mais robusto)
    sorted_features = sorted(feature_robustness.items(), key=lambda x: x[1])
    
    print("\nFeatures ordenadas por robustez (do menos ao mais robusto):")
    for feature, score in sorted_features:
        print(f"  {feature}: {score:.4f}")
    
    # Visualizar as 10 features menos robustas
    plt.figure(figsize=(12, 6))
    features = [x[0] for x in sorted_features[:10]]
    scores = [x[1] for x in sorted_features[:10]]
    
    sns.barplot(x=scores, y=features)
    plt.xlabel('Pontuação de Robustez')
    plt.ylabel('Feature')
    plt.title('Top 10 Features Menos Robustas')
    plt.tight_layout()
    plt.show()

# 4. Examinar impacto das perturbações
# -----------------------------------
if 'perturbation_results' in robustness_results:
    perturbation_results = robustness_results['perturbation_results']
    
    print("\nImpacto das perturbações por magnitude:")
    for magnitude, result in perturbation_results.items():
        if 'accuracy' in result:
            print(f"  Magnitude {magnitude}: Acurácia = {result['accuracy']:.4f}")
    
    # Visualizar degradação de desempenho por magnitude de perturbação
    if len(perturbation_results) > 0:
        magnitudes = []
        accuracies = []
        
        for magnitude, result in sorted(perturbation_results.items(), key=lambda x: float(x[0])):
            if 'accuracy' in result:
                magnitudes.append(float(magnitude))
                accuracies.append(result['accuracy'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(magnitudes, accuracies, marker='o')
        plt.xlabel('Magnitude da Perturbação')
        plt.ylabel('Acurácia')
        plt.title('Degradação de Desempenho com Aumento da Perturbação')
        plt.grid(True)
        plt.show()

# 5. Uso direto da classe RobustnessManager
# ----------------------------------------
# Instanciar o gerenciador de robustez diretamente
robustness_mgr = RobustnessManager(rf_dataset, verbose=True)

# Configurar e executar o teste
custom_robustness_results = robustness_mgr.run_robustness_test(
    # Podemos personalizar os parâmetros do teste
    perturbation_magnitudes=[0.1, 0.2, 0.3],  # Magnitudes personalizadas
    n_perturbations=5,                         # Número de perturbações por feature
    features_to_perturb=None,                  # Todas as features (ou especificar uma lista)
    perturbation_types=['gaussian'],           # Usar apenas perturbação gaussiana
    random_state=42
)

print("\nResultados personalizados do teste de robustez:")
print(f"Pontuação geral de robustez: {custom_robustness_results['overall_robustness_score']:.4f}")

# 6. Comparar robustez entre modelos
# --------------------------------
# Criar DBDataset para o modelo Gradient Boosting
gb_dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=models['gradient_boosting'],
    dataset_name="Gradient Boosting"
)

# Criar experimento para o modelo Gradient Boosting
gb_experiment = Experiment(
    dataset=gb_dataset,
    experiment_type="binary_classification",
    tests=["robustness"],
    feature_subset=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],
    random_state=42
)

# Executar teste de robustez
gb_results = gb_experiment.run_tests(config_name="quick")
gb_robustness = gb_results.results['robustness']['overall_robustness_score']

# Comparar pontuações de robustez
model_robustness = {
    'Random Forest': robustness_results['overall_robustness_score'],
    'Gradient Boosting': gb_robustness
}

print("\nComparação de robustez entre modelos:")
for model, score in model_robustness.items():
    print(f"  {model}: {score:.4f}")

# Visualizar comparação
plt.figure(figsize=(8, 5))
plt.bar(model_robustness.keys(), model_robustness.values())
plt.ylabel('Pontuação de Robustez')
plt.title('Comparação de Robustez entre Modelos')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 7. Entender o impacto da robustez em cada classe
# ----------------------------------------------
# Analisar como as perturbações afetam cada classe separadamente
if 'class_robustness' in robustness_results:
    class_robustness = robustness_results['class_robustness']
    
    print("\nRobustez por classe:")
    for class_label, score in class_robustness.items():
        print(f"  Classe {class_label}: {score:.4f}")
    
    # Visualizar robustez por classe
    plt.figure(figsize=(8, 5))
    plt.bar(class_robustness.keys(), class_robustness.values())
    plt.ylabel('Pontuação de Robustez')
    plt.xlabel('Classe')
    plt.title('Robustez por Classe')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# 8. Gerar e salvar relatório HTML
# ------------------------------
# Criar diretório para relatórios
import os
output_dir = "robustness_reports"
os.makedirs(output_dir, exist_ok=True)

# Gerar relatório para o modelo Random Forest
rf_report_path = os.path.join(output_dir, "random_forest_robustness.html")
experiment.save_html(
    test_type="robustness",
    file_path=rf_report_path,
    model_name="Random Forest"
)

# Gerar relatório para o modelo Gradient Boosting
gb_report_path = os.path.join(output_dir, "gradient_boosting_robustness.html")
gb_experiment.save_html(
    test_type="robustness",
    file_path=gb_report_path,
    model_name="Gradient Boosting"
)

print(f"\nRelatórios HTML salvos em:\n  {rf_report_path}\n  {gb_report_path}")

# 9. Uso avançado: Perturbação manual de dados
# ------------------------------------------
# Demonstrar como perturbar dados manualmente usando o utilitário data_perturber
# Selecionar uma feature para perturbação manual
feature_to_perturb = 'mean radius'

# Obter dados originais para perturbação
original_data = X_test.copy()

# Perturbar manualmente uma feature
perturbed_data = data_perturber.perturb_feature(
    data=original_data,
    feature=feature_to_perturb,
    magnitude=0.2,
    method='gaussian',
    random_state=42
)

# Comparar os valores original e perturbado
comparison = pd.DataFrame({
    'Original': original_data[feature_to_perturb],
    'Perturbado': perturbed_data[feature_to_perturb]
})

# Calcular estatísticas da perturbação
perturbation_stats = {
    'Média do valor original': comparison['Original'].mean(),
    'Média do valor perturbado': comparison['Perturbado'].mean(),
    'Diferença média': (comparison['Perturbado'] - comparison['Original']).mean(),
    'Diferença máxima': (comparison['Perturbado'] - comparison['Original']).abs().max(),
    'Diferença média %': (comparison['Perturbado'] - comparison['Original']).mean() / comparison['Original'].mean() * 100
}

print(f"\nEstatísticas da perturbação manual da feature '{feature_to_perturb}':")
for stat_name, value in perturbation_stats.items():
    print(f"  {stat_name}: {value:.4f}")

# Visualizar distribuição original vs. perturbada
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(comparison['Original'], bins=20, alpha=0.5, label='Original')
plt.hist(comparison['Perturbado'], bins=20, alpha=0.5, label='Perturbado')
plt.xlabel(feature_to_perturb)
plt.ylabel('Frequência')
plt.title('Distribuição: Original vs. Perturbado')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(comparison['Original'], comparison['Perturbado'], alpha=0.5)
plt.plot([comparison['Original'].min(), comparison['Original'].max()], 
         [comparison['Original'].min(), comparison['Original'].max()], 
         'r--')
plt.xlabel('Valor Original')
plt.ylabel('Valor Perturbado')
plt.title('Comparação: Original vs. Perturbado')

plt.tight_layout()
plt.show()
```

## Como Funcionam os Testes de Robustez

Os testes de robustez no DeepBridge avaliam a estabilidade do modelo quando as features de entrada são alteradas ligeiramente. O processo inclui:

1. **Perturbação de Features**: Cada feature é modificada com diferentes níveis de "ruído"
2. **Avaliação do Impacto**: O desempenho do modelo é medido com os dados perturbados
3. **Pontuação de Robustez**: Uma pontuação de 0 a 1 é atribuída a cada feature e ao modelo como um todo

### Tipos de Perturbações

- **Gaussian (Gaussiana)**: Adiciona ruído gaussiano com diferentes magnitudes
- **Uniform (Uniforme)**: Adiciona ruído uniforme dentro de um intervalo
- **Dropout**: Substitui valores por zero com uma probabilidade específica
- **Categorical**: Para features categóricas, altera as categorias

### Configurações do Teste

- **perturbation_magnitudes**: Lista de intensidades das perturbações (ex: [0.1, 0.2, 0.3])
- **n_perturbations**: Número de perturbações por feature
- **features_to_perturb**: Lista específica de features a perturbar (None = todas)
- **perturbation_types**: Tipos de perturbações a serem usadas (ex: ['gaussian', 'uniform'])

### Níveis de Configuração

O DeepBridge oferece três níveis predefinidos de teste de robustez:

1. **quick**: Avaliação rápida com poucas perturbações (bom para exploração inicial)
2. **medium**: Equilíbrio entre velocidade e profundidade
3. **full**: Avaliação abrangente com muitas perturbações (mais demorado)

## Interpretando os Resultados

### Pontuação de Robustez

- Varia de 0 (não robusto) a 1 (extremamente robusto)
- Uma pontuação mais alta indica que o modelo mantém seu desempenho mesmo com perturbações nas features
- Pontuações são fornecidas por feature e globalmente para o modelo

### Resultados por Feature

Obtidos via `feature_robustness` nos resultados, estes valores permitem:
- Identificar quais features causam maior instabilidade no modelo
- Priorizar esforços de engenharia de features
- Refinar o modelo para ser mais robusto em features críticas

### Resultados por Magnitude

Obtidos via `perturbation_results`, mostram:
- Como o desempenho do modelo degrada com o aumento da magnitude das perturbações
- A relação entre a intensidade da perturbação e a queda na acurácia

## Pontos-Chave

- Os testes de robustez avaliam a estabilidade do modelo frente a variações nos dados de entrada
- Um modelo robusto é menos suscetível a pequenas mudanças nas features
- A pontuação de robustez por feature ajuda a identificar pontos fracos do modelo
- Comparar a robustez entre modelos é útil para escolher o modelo mais estável
- Os relatórios HTML oferecem visualizações detalhadas do comportamento do modelo sob perturbações
- A classe `RobustnessManager` pode ser usada diretamente para testes mais personalizados