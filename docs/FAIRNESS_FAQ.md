# FAQ: Fairness no DeepBridge

## 📋 Índice

- [Conceitos Básicos](#conceitos-básicos)
- [Uso do Módulo](#uso-do-módulo)
- [Métricas](#métricas)
- [Interpretação](#interpretação)
- [Mitigação](#mitigação)
- [Questões Técnicas](#questões-técnicas)
- [Questões Legais](#questões-legais)
- [Troubleshooting](#troubleshooting)

---

## Conceitos Básicos

### O que é fairness em Machine Learning?

**Resposta**: Fairness em ML refere-se à ausência de viés ou discriminação injusta contra grupos ou indivíduos baseado em atributos sensíveis (raça, gênero, idade, etc.). Um modelo "justo" toma decisões que não discriminam sistematicamente contra grupos protegidos.

**Exemplo**: Um modelo de aprovação de crédito que aprova homens a uma taxa significativamente maior que mulheres, mesmo quando têm perfis financeiros similares, é considerado injusto.

---

### Qual a diferença entre bias e fairness?

**Resposta**:
- **Bias (Viés)**: Desvio sistemático que favorece ou prejudica certos grupos. Pode ser estatístico ou social.
- **Fairness**: Conceito normativo sobre o que é "justo". Fairness busca mitigar bias social/discriminatório.

**Exemplo**: Um modelo pode ter bias estatístico (regularização) sem problemas de fairness. Mas bias social (prever que mulheres são piores programadoras) é uma questão de fairness.

---

### Por que simplesmente remover atributos sensíveis não funciona?

**Resposta**: Porque existem **proxies** - features correlacionadas com atributos sensíveis que permitem ao modelo "inferir" informações protegidas.

**Exemplo**:
- Remover "raça" do dataset
- Mas manter "CEP" (zip code)
- CEP é altamente correlacionado com raça nos EUA devido a segregação histórica
- Modelo usa CEP como proxy para raça

**Solução**: Análise de correlações + técnicas de mitigação específicas.

---

### Fairness vs. Acurácia: Sempre há trade-off?

**Resposta**: **Nem sempre**, mas frequentemente sim.

**Quando NÃO há trade-off**:
- Se o viés vem de dados ruins/enviesados, corrigir os dados pode AUMENTAR acurácia E fairness
- Se o modelo está overfitting em correlações espúrias

**Quando HÁ trade-off**:
- Quando a distribuição real tem diferenças entre grupos
- Quando otimizar para uma definição de fairness prejudica outra

**Recomendação**: Sempre medir o trade-off. Pequenas perdas em acurácia (1-2%) geralmente são aceitáveis para ganhos significativos em fairness.

---

## Uso do Módulo

### Como começar a usar o módulo de Fairness?

**Resposta**: Há duas formas principais:

#### 1. Via Experiment (Recomendado - Mais Simples)

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Criar dataset
dataset = DBDataset(data=df, target_column='target', model=model)

# Criar experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender', 'race']
)

# Executar testes
result = experiment.run_fairness_tests(config='full')

# Gerar relatório
result.save_html('fairness_report.html', model_name='My Model')
```

#### 2. Via FairnessSuite (Avançado - Mais Controle)

```python
from deepbridge.validation.wrappers import FairnessSuite

# Criar suite
fairness = FairnessSuite(dataset, protected_attributes=['gender', 'race'])

# Executar com configuração específica
results = fairness.config('full').run()

# Gerar relatório manualmente
from deepbridge.core.experiment.report.report_manager import ReportManager
report_manager = ReportManager()
report_manager.generate_report(
    test_type='fairness',
    results=results,
    file_path='fairness_report.html'
)
```

---

### Qual configuração devo usar: quick, medium ou full?

**Resposta**: Depende do estágio do projeto:

| Configuração | Quando Usar | Tempo | Métricas |
|--------------|-------------|-------|----------|
| **quick** | Exploração inicial, protótipos, testes rápidos | 10-30s | 2 métricas |
| **medium** | Validação intermediária, desenvolvimento | 1-3min | 5 pós + 4 pré + CM |
| **full** | Auditoria final, produção, compliance | 5-10min | 11 pós + 4 pré + CM + threshold |

**Recomendação**:
- Desenvolvimento: `quick` → `medium` → `full`
- Produção: Sempre `full`
- Monitoramento contínuo: `medium`

---

### Como especificar atributos protegidos?

**Resposta**:

#### Opção 1: Explícito (RECOMENDADO para produção)

```python
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender', 'race', 'age_group']  # Lista explícita
)
```

#### Opção 2: Auto-detecção (Apenas para exploração)

```python
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"]  # Não especifica - detecta automaticamente
)
# Auto-detecta usando fuzzy matching: 'gender', 'race', 'age', etc.
```

**⚠️ Warning**: Auto-detecção é conveniente mas pode:
- Detectar atributos incorretos
- Perder atributos sensíveis não-óbvios
- Não ser determinístico entre execuções

**Para produção**: SEMPRE especifique explicitamente.

---

### Posso usar com modelos de regressão ou multiclass?

**Resposta**: **Atualmente, apenas classificação binária está suportada** (Fase 1-6 focou em binary classification).

**Suporte planejado futuro**:
- ✅ Classificação Binária (disponível)
- 🔜 Classificação Multiclass (planejado)
- 🔜 Regressão (planejado - métricas diferentes)

**Workaround para multiclass**:
- Converter para one-vs-rest (múltiplas análises binárias)
- Avaliar cada classe separadamente

---

## Métricas

### Quantas métricas de fairness existem?

**Resposta**: O DeepBridge implementa **15 métricas** divididas em:

**Pré-treino** (4 - independentes do modelo):
1. Class Balance (BCL)
2. Concept Balance (BCE)
3. KL Divergence
4. JS Divergence

**Pós-treino** (11 - dependentes do modelo):
1. Statistical Parity (PE)
2. Disparate Impact (ID)
3. Equal Opportunity (IO)
4. Equalized Odds (CP)
5. False Negative Rate Difference (TFN)
6. Conditional Acceptance (TAC)
7. Conditional Rejection (TRJ)
8. Precision Difference (DP)
9. Accuracy Difference (DA)
10. Treatment Equality (IT)
11. Entropy Index (IE)

---

### Qual a diferença entre Statistical Parity e Equal Opportunity?

**Resposta**:

#### Statistical Parity (Paridade Estatística)
- **Fórmula**: P(Ŷ=1 | A=a) - P(Ŷ=1 | A=b)
- **Significado**: Taxa de predições positivas deve ser igual entre grupos
- **Foco**: Resultados iguais (outcome-based)
- **Exemplo**: 50% homens aprovados → 50% mulheres aprovadas

#### Equal Opportunity (Oportunidade Igual)
- **Fórmula**: TPR(A=a) - TPR(A=b)
- **Significado**: Entre indivíduos QUALIFICADOS, taxa de aprovação deve ser igual
- **Foco**: Igualdade para qualificados (merit-based)
- **Exemplo**: Dos homens QUE DEVEM ser aprovados, 80% são → Das mulheres QUE DEVEM ser aprovadas, 80% são

**Quando usar**:
- **Statistical Parity**: Quando queremos resultados proporcionais (recrutamento, admissões)
- **Equal Opportunity**: Quando queremos igualdade para qualificados (crédito, promoções)

---

### O que é Disparate Impact e por que é importante?

**Resposta**:

**Definição**: Razão entre a taxa de outcomes positivos do grupo desfavorecido e do grupo favorecido.

**Fórmula**:
```
Disparate Impact = P(Ŷ=1 | A=desfavorecido) / P(Ŷ=1 | A=favorecido)
```

**Exemplo**:
- 60% dos homens aprovados
- 45% das mulheres aprovadas
- Disparate Impact = 45% / 60% = 0.75

**Importância Legal**:
- **EEOC 80% Rule** (EUA): Disparate Impact < 0.80 é evidência prima facie de discriminação
- Usado em processos de emprego, crédito, habitação

**Interpretação DeepBridge**:
- ✅ Verde: ≥ 0.80 (EEOC compliant)
- ⚠️ Amarelo: 0.70 - 0.79 (atenção)
- ❌ Vermelho: < 0.70 (crítico)

---

### Como são calculadas as métricas pré-treino?

**Resposta**: Métricas pré-treino analisam APENAS os dados, sem considerar o modelo.

#### Class Balance (BCL)
```python
# Diferença na proporção de classes por grupo
P(Y=1 | A=male) - P(Y=1 | A=female)
```

#### Concept Balance (BCE)
```python
# Diferença nas features médias entre grupos (classe positiva)
mean(X | Y=1, A=male) - mean(X | Y=1, A=female)
```

#### KL/JS Divergence
```python
# Divergência entre distribuições de features por grupo
KL(P(X|A=male) || P(X|A=female))
JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)  # M = média
```

**Utilidade**:
- Detectar viés NOS DADOS antes do treinamento
- Independente do modelo
- Útil para diagnóstico inicial

---

### O que significa "Threshold Analysis"?

**Resposta**: Análise de como diferentes thresholds de decisão afetam fairness e performance.

**Como funciona**:
1. Modelo produz probabilidades: `P(Y=1) = 0.65`
2. Threshold converte em decisão: `if P(Y=1) >= 0.5 then Ŷ=1`
3. Threshold Analysis testa 99 valores (0.01 a 0.99)
4. Para cada threshold, calcula: Disparate Impact, Statistical Parity, F1 Score

**Resultado**: Threshold ótimo que maximiza fairness (ou balance com F1)

**Exemplo**:
```
Threshold padrão: 0.50
- Disparate Impact: 0.72 ❌
- F1 Score: 0.82

Threshold ótimo: 0.42
- Disparate Impact: 0.81 ✅
- F1 Score: 0.80 (perda de 2%)
```

**Quando usar**: Quando você pode aceitar pequena perda de performance para ganho significativo em fairness.

---

## Interpretação

### O que é Overall Fairness Score?

**Resposta**: Métrica agregada (0-1) que resume fairness geral do modelo.

**Cálculo**:
```python
# Média ponderada de:
# 1. Métricas pré-treino normalizadas
# 2. Métricas pós-treino normalizadas
# 3. Penalidade por critical issues

score = (
    0.3 * pretrain_score +
    0.7 * posttrain_score -
    0.05 * num_critical_issues
)
```

**Interpretação**:
- **0.90-1.00**: ✅ Excelente
- **0.80-0.89**: ✓ Boa
- **0.70-0.79**: ⚠️ Moderada
- **< 0.70**: ❌ Crítica

**Limitações**:
- Score único esconde nuances
- Sempre revisar métricas individuais
- Considerar contexto específico

---

### Meu modelo tem score 0.65. Posso colocá-lo em produção?

**Resposta**: **NÃO recomendado** sem mitigações.

**Score 0.65 indica**:
- Problemas significativos de fairness
- Provavelmente múltiplos critical issues
- Risco legal e reputacional

**Próximos passos**:
1. Revisar `critical_issues` e `warnings`
2. Identificar métricas específicas problemáticas
3. Aplicar técnicas de mitigação
4. Re-treinar e re-avaliar
5. Só deploy quando score ≥ 0.80

**Exceções** (com documentação legal):
- Contexto de baixo risco (recomendações não-críticas)
- Supervisão humana obrigatória
- Plano claro de melhoria contínua

---

### Como interpretar "Confusion Matrix por Grupo"?

**Resposta**: Mostra a matriz de confusão separadamente para cada grupo demográfico.

**Exemplo**:

| | Male | | Female | |
|---|---|---|---|---|
| | Pred 0 | Pred 1 | Pred 0 | Pred 1 |
| **Real 0** | 850 (TN) | 50 (FP) | 420 (TN) | 80 (FP) |
| **Real 1** | 30 (FN) | 70 (TP) | 40 (FN) | 60 (TP) |

**Métricas derivadas**:
- **Male**: TPR = 70/(70+30) = 70%, FPR = 50/(850+50) = 5.6%
- **Female**: TPR = 60/(60+40) = 60%, FPR = 80/(420+80) = 16%

**Insights**:
- Modelo tem MENOR TPR para mulheres (60% vs 70%) → Perde mais mulheres qualificadas
- Modelo tem MAIOR FPR para mulheres (16% vs 5.6%) → Aprova mais mulheres não-qualificadas erroneamente

**Ação**: Investigar por que modelo performa diferente por grupo.

---

## Mitigação

### Quais técnicas posso usar para mitigar viés?

**Resposta**: Técnicas dividem-se em 3 categorias:

#### 1. Pré-processamento (Antes do Treinamento)

**Re-balanceamento**:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Reweighting**:
```python
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', y)
model.fit(X, y, sample_weight=weights)
```

**Remoção de proxies**:
```python
# Identificar features correlacionadas
high_corr = ['zip_code', 'first_name']
X_clean = X.drop(columns=high_corr)
```

---

#### 2. In-processing (Durante o Treinamento)

**Fairness Constraints** (via Fairlearn):
```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(
    estimator=base_model,
    constraints=DemographicParity()
)
mitigator.fit(X, y, sensitive_features=df['gender'])
```

**Adversarial Debiasing** (via AIF360):
```python
from aif360.algorithms.inprocessing import AdversarialDebiasing
model = AdversarialDebiasing(...)
model.fit(dataset)
```

---

#### 3. Pós-processamento (Após o Treinamento)

**Threshold Optimization**:
```python
# Usar threshold analysis do DeepBridge
result = experiment.run_fairness_tests(config='full')
optimal_threshold = result._results['threshold_analysis']['optimal_threshold']

# Aplicar threshold
y_pred = (model.predict_proba(X)[:, 1] >= optimal_threshold).astype(int)
```

**Calibração por Grupo**:
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar separadamente
for group in groups:
    group_data = X[df['gender'] == group]
    calibrated = CalibratedClassifierCV(model, cv=5)
    calibrated.fit(group_data, y[df['gender'] == group])
```

**Recomendação**: Comece com pré-processamento (mais simples), depois tente in-processing se necessário.

---

### Aplicar técnicas de mitigação sempre melhora fairness?

**Resposta**: **Não necessariamente** - pode ter efeitos colaterais.

**Possíveis problemas**:
1. **Redução de performance geral**: Acurácia pode cair 5-10%
2. **Fairness em uma métrica, bias em outra**: Melhorar Statistical Parity pode piorar Equal Opportunity
3. **Overfitting**: Re-balanceamento excessivo pode causar overfitting no grupo minoritário

**Recomendações**:
- Sempre avaliar ANTES e DEPOIS
- Validar em conjunto de teste independente
- Medir trade-offs explicitamente
- Documentar decisões

**Pipeline recomendado**:
```python
# 1. Baseline
baseline_result = experiment.run_fairness_tests(config='full')

# 2. Aplicar mitigação
# [seu código de mitigação]

# 3. Re-avaliar
mitigated_result = experiment.run_fairness_tests(config='full')

# 4. Comparar
print(f"Baseline: {baseline_result.overall_fairness_score:.3f}")
print(f"Mitigated: {mitigated_result.overall_fairness_score:.3f}")
print(f"Acurácia Baseline: {baseline_acc:.3f}")
print(f"Acurácia Mitigada: {mitigated_acc:.3f}")
```

---

## Questões Técnicas

### Posso usar com qualquer tipo de modelo?

**Resposta**: **Sim**, desde que o modelo tenha interface sklearn-compatible.

**Modelos suportados**:
- ✅ Scikit-learn (RandomForest, LogisticRegression, SVM, etc.)
- ✅ XGBoost
- ✅ LightGBM
- ✅ CatBoost
- ✅ Redes Neurais (Keras/TensorFlow/PyTorch com wrapper sklearn)

**Requisitos**:
1. Método `predict(X)` que retorna classes
2. (Opcional) Método `predict_proba(X)` para threshold analysis

**Exemplo com XGBoost**:
```python
import xgboost as xgb

# Treinar XGBoost
model = xgb.XGBClassifier(...)
model.fit(X_train, y_train)

# Usar com DeepBridge (funciona diretamente)
dataset = DBDataset(data=df, target_column='target', model=model)
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender']
)
```

---

### Como lidar com múltiplos atributos protegidos?

**Resposta**: DeepBridge analisa cada atributo **separadamente**.

**Exemplo**:
```python
protected_attributes = ['gender', 'race', 'age_group']
```

**Análise produzida**:
- Métricas para `gender` (comparando Male vs Female)
- Métricas para `race` (comparando White vs Black vs Hispanic vs Asian)
- Métricas para `age_group` (comparando Young vs Adult vs Middle-Aged vs Senior)

**Limitação atual**: Não analisa **interseções** (ex: Mulheres Negras vs Homens Brancos).

**Workaround para interseções**:
```python
# Criar atributo combinado
df['gender_race'] = df['gender'] + '_' + df['race']
# Resultado: 'Male_White', 'Female_Black', etc.

# Analisar interseção
protected_attributes = ['gender_race']
```

---

### Quanto tempo demora a análise?

**Resposta**: Depende de:
1. Configuração (quick/medium/full)
2. Tamanho do dataset
3. Número de atributos protegidos
4. Número de grupos por atributo

**Benchmarks típicos**:

| Dataset | Config | Atributos | Tempo |
|---------|--------|-----------|-------|
| 1K samples | quick | 2 | ~5s |
| 1K samples | medium | 2 | ~30s |
| 1K samples | full | 2 | ~2min |
| 10K samples | full | 3 | ~5min |
| 100K samples | full | 3 | ~15min |

**Componente mais lento**: Threshold Analysis (testa 99 thresholds)

**Dica de performance**:
```python
# Para datasets grandes, use amostragem
import numpy as np

sample_size = 10000
sample_idx = np.random.choice(len(df), sample_size, replace=False)
df_sample = df.iloc[sample_idx]

# Análise na amostra (muito mais rápido)
dataset = DBDataset(data=df_sample, ...)
```

---

### Os resultados são determinísticos?

**Resposta**: **Sim**, se você controlar seeds.

**Fontes de aleatoriedade**:
1. Split train/test no Experiment
2. Modelo treinado (se usar random_state)
3. Re-balanceamento de dados (SMOTE, etc.)

**Como garantir reprodutibilidade**:
```python
import numpy as np
import random

# Fixar seeds
np.random.seed(42)
random.seed(42)

# Usar random_state no Experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender'],
    random_state=42  # Importante!
)

# Usar random_state no modelo
model = RandomForestClassifier(random_state=42)
```

**Com seeds fixadas**: Resultados são 100% reprodutíveis.

---

## Questões Legais

### Meu modelo é EEOC compliant?

**Resposta**: Verifique a métrica **Disparate Impact**.

**Regra EEOC 80%**:
- Se Disparate Impact ≥ 0.80 para TODOS os atributos protegidos → **Provavelmente compliant**
- Se Disparate Impact < 0.80 para QUALQUER atributo → **Risco de violação**

**No relatório DeepBridge**:
```
Disparate Impact (ID)
  Valor: 0.85
  Interpretação: ✓ Verde (OK)
  EEOC: 0.80 (atende)
```

**⚠️ IMPORTANTE**:
- EEOC 80% é uma **heurística**, não garantia legal
- Compliance real depende de contexto, jurisdição, documentação
- **Sempre consulte advogado** antes de decisões legais

**Documentação recomendada**:
1. Salvar relatório HTML de cada análise
2. Manter logs de todas as decisões
3. Documentar justificativas técnicas
4. Revisar com equipe legal

---

### Quais atributos são legalmente protegidos?

**Resposta**: Varia por jurisdição.

#### Estados Unidos (EEOC)
- Race (raça)
- Color (cor)
- Religion (religião)
- Sex (sexo/gênero)
- National Origin (origem nacional)
- Age (40+) (idade)
- Disability (deficiência)
- Genetic Information (informação genética)

#### Europa (GDPR - Artigo 9)
- Racial/ethnic origin
- Political opinions
- Religious/philosophical beliefs
- Trade union membership
- Genetic data
- Biometric data
- Health data
- Sex life/sexual orientation

#### Brasil (LGPD + Constituição)
- Origem racial/étnica
- Convicção religiosa
- Opinião política
- Filiação sindical
- Dados genéticos
- Dados de saúde
- Orientação sexual
- Dados biométricos

**Recomendação**: Consultar legislação local + advogado especializado.

---

### Posso processar dados sensíveis para análise de fairness?

**Resposta**: **Depende da jurisdição e consentimento**.

#### GDPR (Europa)
- Art. 9: Processamento de dados sensíveis é **proibido por padrão**
- **Exceções**: Consentimento explícito, obrigação legal, interesse público substancial
- **Para fairness**: Geralmente permitido sob "interesse público substancial" (prevenir discriminação)
- **Requisitos**: Documentar necessidade, minimizar dados, garantir segurança

#### LGPD (Brasil)
- Art. 11: Dados sensíveis requerem **consentimento específico**
- **Exceções**: Cumprimento de obrigação legal, exercício regular de direitos
- **Para fairness**: Argumento de "prevenção de discriminação" pode se aplicar
- **Requisitos**: Base legal clara, relatório de impacto

**Recomendações práticas**:
1. **Obter consentimento** quando possível
2. **Minimizar dados**: Apenas atributos necessários
3. **Anonimizar**: Remover identificadores diretos
4. **Documentar**: Justificar necessidade de cada atributo
5. **Limitar acesso**: Apenas equipes autorizadas
6. **Auditoria**: Manter logs de acesso

---

## Troubleshooting

### Erro: "No protected attributes detected"

**Problema**: Auto-detecção não encontrou atributos sensíveis.

**Causas**:
1. Dataset não tem colunas com nomes óbvios ('gender', 'race', etc.)
2. Nomes das colunas muito diferentes dos keywords
3. Atributos protegidos codificados numericamente (0/1 ao invés de 'Male'/'Female')

**Solução**:
```python
# Opção 1: Especificar explicitamente
protected_attributes = ['column_x', 'column_y']  # Use os nomes reais

# Opção 2: Renomear colunas
df = df.rename(columns={'column_x': 'gender', 'column_y': 'race'})

# Opção 3: Ajustar threshold de fuzzy matching
detected = Experiment.detect_sensitive_attributes(dataset, threshold=0.5)  # Mais permissivo
```

---

### Erro: "Feature names mismatch"

**Problema**: Modelo foi treinado com features diferentes das fornecidas.

**Causa comum**: Treinou modelo SEM atributos protegidos, mas dataset inclui atributos protegidos.

**Exemplo do erro**:
```
Modelo treinado com: ['income', 'credit_score', 'debt_ratio']
Dataset fornecido com: ['income', 'credit_score', 'debt_ratio', 'gender', 'race']
```

**Solução**:
```python
# Garantir que DBDataset usa mesmas features do treinamento
feature_cols = ['income', 'credit_score', 'debt_ratio']  # SEM atributos protegidos

# Separar features para predição
X = df[feature_cols]

# Mas manter atributos protegidos no DataFrame completo
dataset = DBDataset(data=df, target_column='target', model=model)
# DeepBridge internamente filtra features corretas
```

---

### Warning: "For production, explicitly specify protected_attributes"

**Problema**: Você está usando auto-detecção.

**Significado**: DeepBridge detectou atributos automaticamente mas não é recomendado para produção.

**Solução**:
```python
# ANTES (auto-detecção)
experiment = Experiment(
    dataset=dataset,
    tests=["fairness"]
)

# DEPOIS (explícito)
experiment = Experiment(
    dataset=dataset,
    tests=["fairness"],
    protected_attributes=['gender', 'race']  # Adicionar explicitamente
)
```

---

### Overall Fairness Score muito baixo mas visual parece OK

**Problema**: Score agregado pode ser enganoso.

**Causa**: Score penaliza QUALQUER métrica crítica, mesmo se maioria está OK.

**Exemplo**:
```
Overall Score: 0.65 (parece crítico)

Métricas individuais:
- Statistical Parity: ✓ 0.08 (OK)
- Equal Opportunity: ✓ 0.06 (OK)
- Equalized Odds: ✓ 0.09 (OK)
- Disparate Impact: ✗ 0.65 (crítico!) <- Puxa score para baixo
```

**Solução**:
1. **Não confiar apenas no Overall Score**
2. Revisar `critical_issues` para identificar problema específico
3. Focar mitigação na métrica problemática
4. Considerar contexto (algumas métricas são mais importantes que outras)

---

### Relatório HTML não abre / caracteres estranhos

**Problema**: Encoding UTF-8 não reconhecido.

**Solução**:
```python
# Ao salvar, garantir encoding
result.save_html('report.html', model_name='Model')

# Ao abrir manualmente, especificar encoding
with open('report.html', 'r', encoding='utf-8') as f:
    content = f.read()
```

**Navegadores**: Todos os navegadores modernos (Chrome, Firefox, Safari, Edge) suportam UTF-8 por padrão.

---

### Análise muito lenta (> 20 minutos)

**Problema**: Dataset muito grande ou muitos atributos/grupos.

**Soluções**:

#### 1. Usar configuração mais leve
```python
# Ao invés de 'full', usar 'medium' ou 'quick'
result = experiment.run_fairness_tests(config='medium')
```

#### 2. Amostragem estratificada
```python
from sklearn.model_selection import train_test_split

# Amostrar 10K samples mantendo proporções
df_sample, _ = train_test_split(
    df,
    train_size=10000,
    stratify=df['target'],
    random_state=42
)

dataset = DBDataset(data=df_sample, ...)
```

#### 3. Reduzir atributos protegidos
```python
# Analisar um atributo por vez
for attr in ['gender', 'race', 'age_group']:
    result = experiment.run_fairness_tests(
        protected_attributes=[attr],  # Um por vez
        config='full'
    )
    result.save_html(f'fairness_{attr}.html')
```

---

## Recursos Adicionais

### Onde encontrar mais informações?

- **Documentação Completa**: `docs/FAIRNESS_BEST_PRACTICES.md`
- **Tutorial Passo-a-Passo**: `docs/FAIRNESS_TUTORIAL.md`
- **Exemplo Completo**: `examples/fairness_complete_example.py`
- **Código Fonte**: `deepbridge/validation/fairness/`

### Bibliotecas Complementares

- **AIF360** (IBM): https://github.com/Trusted-AI/AIF360
- **Fairlearn** (Microsoft): https://fairlearn.org/
- **What-If Tool** (Google): https://pair-code.github.io/what-if-tool/

### Artigos Acadêmicos Recomendados

1. **"Fairness and Machine Learning"** - Barocas, Hardt, Narayanan (2019)
2. **"A Survey on Bias and Fairness in Machine Learning"** - Mehrabi et al. (2021)
3. **"Fairness Definitions Explained"** - Verma & Rubin (2018)

---

## Ainda tem dúvidas?

**Reporte issues**: https://github.com/[seu-repo]/DeepBridge/issues
**Contribua**: Pull requests são bem-vindos!

---

**Versão**: 1.0
**Última atualização**: 2025-11-03
