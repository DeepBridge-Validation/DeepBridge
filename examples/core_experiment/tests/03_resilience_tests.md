# Testes de Resiliência

Este exemplo demonstra como realizar testes de resiliência de modelos usando o módulo `core.experiment`. Os testes de resiliência avaliam como o modelo se comporta quando a distribuição dos dados muda (drift).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.managers.resilience_manager import ResilienceManager

# 1. Criar dados com possibilidade de drift controlado
# --------------------------------------------------
# Função para gerar dados com drift
def generate_data_with_drift(n_samples=1000, n_features=10, shift_factor=0):
    """
    Gera dados para classificação com possibilidade de drift controlado.
    
    Args:
        n_samples: Número de amostras
        n_features: Número de features
        shift_factor: Fator de mudança na distribuição (0 = sem mudança)
        
    Returns:
        X, y: Features e target
    """
    # Gerar dados base
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features-2,
        n_redundant=2,
        random_state=42
    )
    
    # Aplicar shift na distribuição se fator diferente de zero
    if shift_factor != 0:
        # Adicionar shift às features informativas
        for i in range(n_features-2):
            # Mudança na média
            X[:, i] += shift_factor
            # Mudança na variância
            X[:, i] *= (1 + 0.2 * shift_factor)
    
    # Converter para DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

# Gerar dados de treino (sem drift)
X_train, y_train = generate_data_with_drift(n_samples=1000, n_features=10, shift_factor=0)

# Gerar dados de teste com diferentes níveis de drift
test_data = {}
for shift in [0, 0.5, 1.0, 1.5]:
    X_test, y_test = generate_data_with_drift(n_samples=300, n_features=10, shift_factor=shift)
    test_data[f'shift_{shift}'] = (X_test, y_test)

print(f"Dados de treino: {X_train.shape}")
for name, (X, y) in test_data.items():
    print(f"Dados de teste ({name}): {X.shape}")

# 2. Treinar diferentes modelos para comparação
# -------------------------------------------
models = {
    'logistic_regression': LogisticRegression(random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Treinar cada modelo nos dados de treino
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Modelo {name} treinado")

# 3. Configurar e executar teste de resiliência via Experiment
# ----------------------------------------------------------
# Criar DataFrame de treino
train_df = pd.concat([X_train, y_train], axis=1)

# Desempenho nos diferentes conjuntos de teste (baseline para comparação)
print("\nDesempenho de base nos diferentes conjuntos de teste:")
baseline_performance = {}

for model_name, model in models.items():
    baseline_performance[model_name] = {}
    
    for test_name, (X_test, y_test) in test_data.items():
        accuracy = model.score(X_test, y_test)
        baseline_performance[model_name][test_name] = accuracy
        print(f"  {model_name} em {test_name}: {accuracy:.4f}")

# Executar testes de resiliência para cada modelo
resilience_results = {}

for model_name, model in models.items():
    print(f"\nExecutando teste de resiliência para: {model_name}")
    
    # Usar o conjunto de teste sem drift como conjunto de teste padrão
    X_std_test, y_std_test = test_data['shift_0']
    test_df = pd.concat([X_std_test, y_std_test], axis=1)
    
    # Criar DBDataset
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model,
        dataset_name=model_name
    )
    
    # Criar experimento focado em testes de resiliência
    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["resilience"],  # Apenas teste de resiliência
        random_state=42
    )
    
    # Executar teste de resiliência
    results = experiment.run_tests(config_name="quick")
    
    # Armazenar resultados
    if 'resilience' in results.results:
        resilience_results[model_name] = results.results['resilience']

# 4. Analisar pontuações de resiliência
# -----------------------------------
# Extrair pontuações de resiliência
resilience_scores = {}
for model_name, results in resilience_results.items():
    if 'overall_resilience_score' in results:
        resilience_scores[model_name] = results['overall_resilience_score']

# Criar DataFrame para comparação
scores_df = pd.DataFrame({
    'Modelo': list(resilience_scores.keys()),
    'Pontuação de Resiliência': list(resilience_scores.values())
})

print("\nPontuações de resiliência por modelo:")
print(scores_df)

# Visualizar pontuações
plt.figure(figsize=(10, 6))
sns.barplot(x='Modelo', y='Pontuação de Resiliência', data=scores_df)
plt.title('Resiliência de Modelos a Mudanças na Distribuição')
plt.ylim(0, 1)
plt.show()

# 5. Analisar resiliência por feature
# ---------------------------------
# Extrair pontuações de resiliência por feature
feature_resilience = {}
for model_name, results in resilience_results.items():
    if 'feature_resilience' in results:
        feature_resilience[model_name] = results['feature_resilience']

# Comparar a resiliência das features entre os modelos
if feature_resilience:
    # Selecionar algumas features para visualização
    selected_features = list(feature_resilience[model_name].keys())[:5]  # Primeiras 5 features
    
    # Criar DataFrame para visualização
    feature_data = []
    for model_name, features in feature_resilience.items():
        for feature, score in features.items():
            if feature in selected_features:
                feature_data.append({
                    'Modelo': model_name,
                    'Feature': feature,
                    'Pontuação': score
                })
    
    feature_df = pd.DataFrame(feature_data)
    
    # Visualizar
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Feature', y='Pontuação', hue='Modelo', data=feature_df)
    plt.title('Resiliência por Feature e Modelo')
    plt.ylim(0, 1)
    plt.legend(title='Modelo')
    plt.show()

# 6. Uso direto da classe ResilienceManager
# ---------------------------------------
# Instanciar o gerenciador de resiliência diretamente (com o primeiro modelo)
model = models['random_forest']
resilience_mgr = ResilienceManager(
    dataset=DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model
    ),
    verbose=True
)

# Configurar e executar o teste de resiliência manualmente
custom_resilience_results = resilience_mgr.run_resilience_test(
    # Parâmetros personalizados
    n_perturbations=5,             # Número de perturbações
    perturbation_types=['shift'],  # Tipos de perturbação
    shift_magnitudes=[0.2, 0.5, 1.0],  # Magnitudes do shift
    random_state=42
)

print("\nResultados personalizados do teste de resiliência:")
print(f"Pontuação geral: {custom_resilience_results['overall_resilience_score']:.4f}")

# 7. Analisar impacto de diferentes tipos de drift
# ---------------------------------------------
# Criar dados com diferentes tipos de shift
drift_types = {
    'No Drift': 0,
    'Mean Shift': 0.8,
    'Variance Shift': 1.2,
    'Combined Shift': 1.5
}

# Avaliar o desempenho do modelo em diferentes cenários de drift
drift_performance = {}

for model_name, model in models.items():
    drift_performance[model_name] = {}
    
    for drift_name, shift_factor in drift_types.items():
        # Gerar dados com o tipo de drift específico
        if drift_name == 'Variance Shift':
            # Criar um shift principalmente na variância
            X_drift, y_drift = generate_data_with_drift(n_samples=300, n_features=10, shift_factor=0)
            # Aplicar manualmente um shift na variância
            for i in range(8):
                X_drift.iloc[:, i] *= (1 + 0.3 * shift_factor)
        else:
            # Shift padrão (afeta média e variância)
            X_drift, y_drift = generate_data_with_drift(n_samples=300, n_features=10, shift_factor=shift_factor)
        
        # Avaliar desempenho
        accuracy = model.score(X_drift, y_drift)
        drift_performance[model_name][drift_name] = accuracy

# Criar DataFrame para visualização
drift_data = []
for model_name, drifts in drift_performance.items():
    for drift_name, accuracy in drifts.items():
        drift_data.append({
            'Modelo': model_name,
            'Tipo de Drift': drift_name,
            'Acurácia': accuracy
        })

drift_df = pd.DataFrame(drift_data)

# Visualizar degradação de desempenho
plt.figure(figsize=(12, 8))
sns.barplot(x='Tipo de Drift', y='Acurácia', hue='Modelo', data=drift_df)
plt.title('Impacto de Diferentes Tipos de Drift no Desempenho do Modelo')
plt.ylim(0.5, 1.0)
plt.legend(title='Modelo')
plt.show()

# 8. Visualizar distribuições antes e depois do drift
# ------------------------------------------------
# Selecionar uma feature para visualização
feature_to_plot = 'feature_0'

plt.figure(figsize=(12, 8))

# Dados de treino (referência)
sns.kdeplot(X_train[feature_to_plot], label='Treino (Referência)', color='black')

# Dados com diferentes níveis de drift
colors = ['blue', 'green', 'orange', 'red']
for i, (name, (X, _)) in enumerate(test_data.items()):
    sns.kdeplot(X[feature_to_plot], label=name, color=colors[i])

plt.title(f'Distribuição de {feature_to_plot} com Diferentes Níveis de Drift')
plt.xlabel(feature_to_plot)
plt.ylabel('Densidade')
plt.legend()
plt.show()

# 9. Gerar e salvar relatórios HTML
# ------------------------------
# Criar diretório para relatórios
import os
output_dir = "resilience_reports"
os.makedirs(output_dir, exist_ok=True)

# Para o modelo com melhor pontuação de resiliência
best_model_name = max(resilience_scores, key=resilience_scores.get)
best_model = models[best_model_name]

# Recriar experimento para o melhor modelo
best_dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=best_model,
    dataset_name=f"{best_model_name} (mais resiliente)"
)

best_experiment = Experiment(
    dataset=best_dataset,
    experiment_type="binary_classification",
    tests=["resilience"],
    random_state=42
)

# Executar teste novamente
best_experiment.run_tests(config_name="medium")  # Usar configuração mais detalhada para o relatório

# Gerar e salvar relatório
report_path = os.path.join(output_dir, f"{best_model_name.lower()}_resilience.html")
best_experiment.save_html(
    test_type="resilience",
    file_path=report_path,
    model_name=f"{best_model_name} (mais resiliente)"
)

print(f"\nRelatório de resiliência para o melhor modelo ({best_model_name}) salvo em: {report_path}")

# 10. Análise Avançada: Avaliar janela de desempenho seguro
# -------------------------------------------------------
# Calcular até que ponto de drift cada modelo mantém um desempenho aceitável

# Definir limite de acurácia aceitável (por exemplo, 85% da acurácia original)
acceptance_threshold = 0.85

# Gerar dados com uma faixa completa de níveis de drift
drift_levels = np.linspace(0, 2.0, 11)  # 0 a 2.0 em 11 pontos
performance_window = {}

for model_name, model in models.items():
    base_accuracy = model.score(X_train, y_train)
    min_acceptable = base_accuracy * acceptance_threshold
    performance_window[model_name] = {'accuracies': [], 'base': base_accuracy, 'acceptable_limit': []}
    
    max_safe_drift = None
    
    for drift in drift_levels:
        X_drift, y_drift = generate_data_with_drift(n_samples=300, n_features=10, shift_factor=drift)
        accuracy = model.score(X_drift, y_drift)
        performance_window[model_name]['accuracies'].append(accuracy)
        performance_window[model_name]['acceptable_limit'].append(min_acceptable)
        
        # Registrar o último nível de drift aceitável
        if accuracy >= min_acceptable and (max_safe_drift is None or drift > max_safe_drift):
            max_safe_drift = drift
    
    performance_window[model_name]['max_safe_drift'] = max_safe_drift

# Visualizar janela de desempenho seguro
plt.figure(figsize=(12, 8))

for model_name, data in performance_window.items():
    plt.plot(drift_levels, data['accuracies'], marker='o', label=model_name)
    plt.plot(drift_levels, data['acceptable_limit'], '--', color='gray', alpha=0.5)
    
    # Marcar o ponto máximo de drift seguro
    if data['max_safe_drift'] is not None:
        idx = list(drift_levels).index(min(drift_levels, key=lambda x: abs(x - data['max_safe_drift'])))
        plt.axvline(x=drift_levels[idx], color='gray', linestyle=':', alpha=0.3)
        plt.text(drift_levels[idx], 0.55, f"{model_name}: {drift_levels[idx]:.1f}", rotation=90)

plt.xlabel('Nível de Drift')
plt.ylabel('Acurácia')
plt.title('Janela de Desempenho Seguro por Modelo')
plt.axhline(y=performance_window[model_name]['base'] * acceptance_threshold, color='red', linestyle='--', label='Limite aceitável')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir os níveis máximos de drift seguro para cada modelo
print("\nNível máximo de drift seguro por modelo:")
for model_name, data in performance_window.items():
    print(f"  {model_name}: {data['max_safe_drift']:.2f}")
```

## Como Funcionam os Testes de Resiliência

Os testes de resiliência no DeepBridge avaliam a capacidade de um modelo de manter seu desempenho quando a distribuição dos dados muda. Este tipo de mudança de distribuição, chamado de "data drift" ou "concept drift", é comum em ambientes do mundo real.

Os testes envolvem:

1. **Perturbação Sistemática**: Introdução controlada de diferentes tipos de drift
2. **Avaliação do Impacto**: Medida da degradação do desempenho sob diferentes perturbações
3. **Pontuação de Resiliência**: Quantificação da robustez do modelo contra mudanças na distribuição

### Tipos de Drift Simulados

- **Shift na Média**: Deslocamento da distribuição inteira (ex: mudança de clima)
- **Shift na Variância**: Mudança na dispersão dos dados (ex: maior variabilidade)
- **Drift nas Correlações**: Mudança nas relações entre variáveis
- **Outros Tipos**: Mudanças na forma da distribuição, aparecimento de outliers, etc.

### Configurações do Teste

- **n_perturbations**: Número de perturbações a serem aplicadas
- **perturbation_types**: Tipos de perturbação a serem usados ('shift', 'scale', etc.)
- **shift_magnitudes**: Intensidades dos shifts a serem aplicados
- **random_state**: Semente para reprodutibilidade

## Importância dos Testes de Resiliência

Em aplicações do mundo real, os dados raramente permanecem estáticos:

- **Evolução Temporal**: Padrões mudam ao longo do tempo
- **Diferentes Contextos**: Aplicação do modelo em novos domínios
- **Fontes de Dados**: Mudanças no hardware, sensores ou métodos de coleta
- **Comportamentos**: Adaptação dos usuários ao sistema

Modelos resilientes são capazes de:

1. **Manter Desempenho**: Continuar funcionando aceitavelmente sob drift moderado
2. **Degradar Graciosamente**: Perder desempenho de forma previsível e controlada
3. **Sinalizar Problemas**: Potencialmente detectar quando estão operando fora de sua zona de confiança

## Pontuação de Resiliência

- Varia de 0 (não resiliente) a 1 (extremamente resiliente)
- Pontuações mais altas indicam que o modelo mantém seu desempenho mesmo com mudanças na distribuição
- Pontuações são fornecidas globalmente e por feature

## Interpretando os Resultados

### Pontuação Geral de Resiliência

- **Alta (>0.8)**: Modelo muito resiliente a mudanças na distribuição
- **Média (0.5-0.8)**: Resiliência moderada, vigilância necessária
- **Baixa (<0.5)**: Modelo sensível a mudanças, requer monitoramento frequente

### Resiliência por Feature

- Identifica quais features são mais sensíveis a mudanças
- Ajuda a priorizar esforços de engenharia e monitoramento
- Útil para entender os pontos fracos do modelo

### Janela de Desempenho Seguro

- Define os limites dentro dos quais o modelo ainda é confiável
- Ajuda a estabelecer limites para reentreinar ou revisar o modelo
- Fornece expectativas realistas sobre a longevidade do modelo

## Aplicações Práticas

1. **Seleção de Modelo**: Escolher modelos mais resilientes para ambientes dinâmicos
2. **Monitoramento**: Estabelecer limites de alerta para drift em produção
3. **Manutenção**: Definir políticas de reentreino baseadas em limites de drift
4. **Engenharia de Features**: Identificar e melhorar features com baixa resiliência

## Pontos-Chave

- A resiliência é uma propriedade crítica para modelos em ambientes dinâmicos
- Testes de resiliência simulam diferentes tipos de mudanças na distribuição
- A pontuação de resiliência quantifica a robustez do modelo contra estas mudanças
- Modelos diferentes podem ter perfis de resiliência distintos
- A janela de desempenho seguro ajuda a definir limites operacionais para o modelo
- Os relatórios HTML fornecem visualizações detalhadas do comportamento do modelo sob diferentes condições