# Testes de Incerteza

Este exemplo demonstra como realizar testes de incerteza em modelos de machine learning usando o módulo `core.experiment`. Os testes de incerteza avaliam quão bem um modelo expressa sua própria incerteza nas previsões.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.managers.uncertainty_manager import UncertaintyManager

# 1. Preparar dados e modelos
# --------------------------
# Carregar dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelos para comparação
# Modelo 1: Random Forest padrão
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Modelo 2: Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Modelo 3: Random Forest calibrado com Isotonic Regression
rf_calibrated = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=100, random_state=42),
    cv=5,
    method='isotonic'
)
rf_calibrated.fit(X_train, y_train)

print(f"Modelos treinados")

# 2. Configurar e executar teste de incerteza via Experiment
# --------------------------------------------------------
# Criar DataFrames de treino e teste
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Dicionário para armazenar resultados
uncertainty_results = {}

# Para cada modelo
for model_name, model in [
    ('Random Forest', rf_model), 
    ('Gradient Boosting', gb_model), 
    ('RF Calibrado', rf_calibrated)
]:
    print(f"\nExecutando teste de incerteza para: {model_name}")
    
    # Criar DBDataset
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=model,
        dataset_name=model_name
    )
    
    # Criar experimento focado em testes de incerteza
    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["uncertainty"],  # Apenas teste de incerteza
        random_state=42
    )
    
    # Executar teste de incerteza
    # Configurações: "quick", "medium", "full"
    results = experiment.run_tests(config_name="quick")
    
    # Armazenar resultados
    if 'uncertainty' in results.results:
        uncertainty_results[model_name] = results.results['uncertainty']

# 3. Comparar métricas de calibração entre modelos
# ----------------------------------------------
# Métricas principais para comparar
metrics_to_compare = [
    'calibration_error',  # Erro de calibração (menor é melhor)
    'expected_calibration_error',  # ECE (menor é melhor)
    'brier_score',  # Pontuação de Brier (menor é melhor)
    'sharpness',  # Nitidez das probabilidades (maior é melhor)
    'coverage_error'  # Erro de cobertura dos intervalos de confiança (menor é melhor)
]

# Criar DataFrame para comparação
comparison_df = pd.DataFrame(index=uncertainty_results.keys(), columns=metrics_to_compare)

for model_name, results in uncertainty_results.items():
    for metric in metrics_to_compare:
        if metric in results:
            comparison_df.loc[model_name, metric] = results[metric]

# Mostrar comparação
print("\nComparação de métricas de incerteza entre modelos:")
print(comparison_df.round(4))

# 4. Visualizar curvas de calibração
# --------------------------------
# Função para plotar curvas de calibração
def plot_calibration_curve(uncertainty_results):
    plt.figure(figsize=(10, 6))
    
    for model_name, results in uncertainty_results.items():
        if 'calibration_curve' in results:
            calib_data = results['calibration_curve']
            if 'pred_probs' in calib_data and 'true_probs' in calib_data:
                pred_probs = calib_data['pred_probs']
                true_probs = calib_data['true_probs']
                
                plt.plot(pred_probs, true_probs, marker='o', label=f"{model_name}")
    
    # Linha diagonal (calibração perfeita)
    plt.plot([0, 1], [0, 1], 'k--', label='Calibração Perfeita')
    
    plt.xlabel('Probabilidade Prevista')
    plt.ylabel('Frequência Observada')
    plt.title('Curvas de Calibração - Comparação de Modelos')
    plt.legend(loc='lower right')
    plt.grid(True)
    return plt

# Plotar curvas de calibração
calibration_plot = plot_calibration_curve(uncertainty_results)
calibration_plot.show()

# 5. Visualizar histogramas de probabilidades
# -----------------------------------------
# Mostrar a distribuição das probabilidades preditas
plt.figure(figsize=(12, 5))

# Para cada modelo
for i, (model_name, results) in enumerate(uncertainty_results.items()):
    if 'confidence_distribution' in results:
        confidence_dist = results['confidence_distribution']
        if isinstance(confidence_dist, list):
            plt.subplot(1, len(uncertainty_results), i+1)
            plt.hist(confidence_dist, bins=20, alpha=0.7)
            plt.title(f'Distribuição de Confiança\n{model_name}')
            plt.xlabel('Probabilidade Prevista')
            plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# 6. Uso direto da classe UncertaintyManager
# ----------------------------------------
# Instanciar o gerenciador de incerteza diretamente (com o primeiro modelo)
uncertainty_mgr = UncertaintyManager(
    dataset=DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=rf_model
    ),
    verbose=True
)

# Configurar e executar o teste de incerteza manualmente
custom_uncertainty_results = uncertainty_mgr.run_uncertainty_test(
    # Parâmetros personalizados
    confidence_thresholds=[0.6, 0.7, 0.8, 0.9, 0.95],  # Personalizar níveis de confiança
    n_bins=15,                                         # Número de bins para calibração
    random_state=42
)

print("\nResultados personalizados do teste de incerteza:")
for metric in ['calibration_error', 'expected_calibration_error', 'sharpness']:
    if metric in custom_uncertainty_results:
        print(f"  {metric}: {custom_uncertainty_results[metric]:.4f}")

# 7. Analisar a relação entre confiança e acurácia
# ----------------------------------------------
# Para cada modelo, plotar a acurácia em diferentes níveis de confiança
plt.figure(figsize=(12, 6))

for model_name, results in uncertainty_results.items():
    if 'reliability_curve' in results:
        reliability = results['reliability_curve']
        if isinstance(reliability, dict) and 'confidence_levels' in reliability and 'accuracies' in reliability:
            confidence_levels = reliability['confidence_levels']
            accuracies = reliability['accuracies']
            
            plt.plot(confidence_levels, accuracies, marker='o', label=model_name)

plt.plot([0, 1], [0, 1], 'k--', label='Calibração Perfeita')
plt.xlabel('Nível de Confiança')
plt.ylabel('Acurácia Observada')
plt.title('Relação entre Confiança e Acurácia')
plt.legend()
plt.grid(True)
plt.show()

# 8. Analisar a cobertura dos intervalos de predição
# ------------------------------------------------
# Para modelos que fornecem intervalos de predição, verificar a cobertura real
plt.figure(figsize=(10, 6))

for model_name, results in uncertainty_results.items():
    if 'prediction_intervals' in results:
        intervals = results['prediction_intervals']
        if isinstance(intervals, dict) and 'expected_coverage' in intervals and 'actual_coverage' in intervals:
            expected = intervals['expected_coverage']
            actual = intervals['actual_coverage']
            
            # Organizar em DataFrame para visualização
            coverage_df = pd.DataFrame({
                'Esperado': expected,
                'Atual': actual
            })
            
            plt.plot(coverage_df['Esperado'], coverage_df['Atual'], marker='o', label=model_name)

plt.plot([0, 1], [0, 1], 'k--', label='Cobertura Ideal')
plt.xlabel('Cobertura Esperada')
plt.ylabel('Cobertura Observada')
plt.title('Análise de Cobertura dos Intervalos de Predição')
plt.legend()
plt.grid(True)
plt.show()

# 9. Gerar e salvar relatórios HTML
# -------------------------------
# Criar diretório para relatórios
import os
output_dir = "uncertainty_reports"
os.makedirs(output_dir, exist_ok=True)

# Para cada modelo, gerar relatório HTML
for model_name in uncertainty_results.keys():
    # Recuperar o experimento correspondente (precisamos recriar, já que não salvamos anteriormente)
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        model=[m for name, m in [('Random Forest', rf_model), 
                               ('Gradient Boosting', gb_model), 
                               ('RF Calibrado', rf_calibrated)] 
               if name == model_name][0],
        dataset_name=model_name
    )
    
    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["uncertainty"],
        random_state=42
    )
    
    # Executar o teste (novamente)
    experiment.run_tests(config_name="quick")
    
    # Gerar e salvar relatório
    report_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_uncertainty.html")
    experiment.save_html(
        test_type="uncertainty",
        file_path=report_path,
        model_name=model_name
    )
    
    print(f"Relatório para {model_name} salvo em: {report_path}")

# 10. Análise de Rejeição - Como o desempenho melhora ao rejeitar predições incertas
# -------------------------------------------------------------------------------
# Demonstrar como a acurácia melhora quando rejeitamos amostras com baixa confiança
def evaluate_rejection_performance(model, X, y, thresholds):
    # Obter probabilidades
    probs = model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    y_pred = model.predict(X)
    
    performance = []
    for threshold in thresholds:
        # Selecionar apenas predições com confiança acima do limite
        mask = max_probs >= threshold
        
        # Se não houver amostras, continuar
        if np.sum(mask) == 0:
            performance.append({
                'threshold': threshold,
                'accuracy': np.nan,
                'coverage': 0,
                'rejection_rate': 1.0
            })
            continue
        
        # Calcular acurácia nas amostras retidas
        accuracy = np.mean(y_pred[mask] == y.values[mask])
        coverage = np.sum(mask) / len(y)
        rejection_rate = 1 - coverage
        
        performance.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'coverage': coverage,
            'rejection_rate': rejection_rate
        })
    
    return pd.DataFrame(performance)

# Definir limites de confiança para teste
confidence_thresholds = np.linspace(0, 0.95, 20)

# Avaliar rejeição para cada modelo
rejection_results = {}
for model_name, model in [
    ('Random Forest', rf_model), 
    ('Gradient Boosting', gb_model), 
    ('RF Calibrado', rf_calibrated)
]:
    rejection_results[model_name] = evaluate_rejection_performance(
        model, X_test, y_test, confidence_thresholds
    )

# Visualizar como a acurácia aumenta com a rejeição
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for model_name, df in rejection_results.items():
    plt.plot(df['rejection_rate'], df['accuracy'], marker='o', label=model_name)
plt.xlabel('Taxa de Rejeição')
plt.ylabel('Acurácia')
plt.title('Acurácia vs. Taxa de Rejeição')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
for model_name, df in rejection_results.items():
    plt.plot(df['threshold'], df['accuracy'], marker='o', label=model_name)
plt.xlabel('Limite de Confiança')
plt.ylabel('Acurácia')
plt.title('Acurácia vs. Limite de Confiança')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

## Como Funcionam os Testes de Incerteza

Os testes de incerteza no DeepBridge avaliam quão bem um modelo expressa sua própria incerteza nas previsões. Os principais aspectos avaliados são:

1. **Calibração**: As probabilidades previstas pelo modelo correspondem às frequências observadas?
2. **Sharpness (Nitidez)**: Quão definidas ou concentradas são as distribuições de probabilidade?
3. **Cobertura dos Intervalos de Predição**: Os intervalos de confiança cobrem a proporção esperada de casos?

### Métricas de Incerteza

- **Calibration Error (Erro de Calibração)**: Mede o desvio entre probabilidades previstas e frequências observadas
- **Expected Calibration Error (ECE)**: Média ponderada do erro de calibração por bins
- **Brier Score**: Mede a precisão das probabilidades previstas (menor é melhor)
- **Sharpness**: Avalia a concentração das distribuições de probabilidade (maior é melhor)
- **Coverage Error**: Diferença entre a cobertura esperada e real dos intervalos de confiança

### Configurações do Teste

- **confidence_thresholds**: Níveis de confiança para avaliar (ex: [0.7, 0.8, 0.9, 0.95])
- **n_bins**: Número de bins para calcular as curvas de calibração
- **config_name**: "quick", "medium", ou "full" (predefinições que controlam a profundidade do teste)

## Importância da Calibração de Probabilidades

Um modelo bem calibrado deve ter as seguintes propriedades:

- **Honestidade**: Quando o modelo prevê uma probabilidade de 80%, aproximadamente 80% dessas previsões devem ser corretas
- **Informatividade**: As probabilidades devem ser concentradas (não uniformes) para serem úteis
- **Representatividade**: O modelo deve expressar corretamente sua incerteza em diferentes regiões do espaço de features

## Aplicações Práticas dos Testes de Incerteza

1. **Tomada de Decisão Baseada em Risco**: Rejeitar predições de baixa confiança
2. **Melhoria de Modelos**: Identificar onde o modelo está mal calibrado
3. **Comparação de Modelos**: Escolher modelos que expressam melhor sua incerteza
4. **Garantia de Qualidade**: Assegurar que as probabilidades do modelo sejam confiáveis

## Interpretando os Resultados

### Curvas de Calibração

- **Linha Diagonal**: Representa calibração perfeita
- **Acima da Diagonal**: Modelo é "pouco confiante" (probabilidades subestimadas)
- **Abaixo da Diagonal**: Modelo é "confiante demais" (probabilidades superestimadas)

### Análise de Rejeição

- Mostra como a acurácia melhora ao rejeitar predições de baixa confiança
- Útil para estabelecer limites de confiança em aplicações críticas
- Ajuda a encontrar o equilíbrio entre acurácia e cobertura

## Métodos de Calibração

Quando um modelo apresenta problemas de calibração, pode-se aplicar técnicas como:

1. **Calibração de Platt**: Aplica regressão logística às saídas do modelo
2. **Calibração Isotônica**: Usa regressão isotônica, mais flexível que a calibração de Platt
3. **Temperatura de Softmax**: Ajusta a "temperatura" das probabilidades de saída

## Pontos-Chave

- Os testes de incerteza avaliam a qualidade das probabilidades previstas pelo modelo
- Um modelo bem calibrado é essencial para aplicações que dependem de estimativas de confiança
- A calibração pode ser visualizada e avaliada quantitativamente
- A classe `UncertaintyManager` permite personalizar os testes de incerteza
- Os relatórios HTML fornecem visualizações detalhadas das propriedades de incerteza do modelo
- A análise de rejeição mostra como usar estimativas de incerteza na prática