# ⚖️ Fairness e Compliance

Aprenda a detectar, quantificar e mitigar bias em modelos de Machine Learning.

<div style="background-color: #fff3e0; padding: 15px; border-radius: 5px;">
<b>⚠️ CRÍTICO:</b> Para aplicações que afetam pessoas (credit, hiring, healthcare, justice), análise de fairness é OBRIGATÓRIA por lei!
</div>

---

## 📓 Notebooks desta Pasta

| # | Notebook | Tempo | Descrição | Prioridade |
|---|----------|-------|-----------|------------|
| 1 | `01_introducao_fairness.ipynb` | 20 min | Conceitos, atributos protegidos, métricas | 🔴 ALTA |
| 2 | `02_analise_completa_fairness.ipynb` ⭐⭐ | 35 min | **CRÍTICO** - 15 métricas + EEOC compliance | 🔴 ALTA |
| 3 | `03_mitigacao_bias.ipynb` | 25 min | Técnicas de correção de bias | 🟡 MÉDIA |

**Tempo Total**: ~80 minutos

---

## 🎯 Ordem Recomendada

### Para Todos (OBRIGATÓRIO se trabalha com modelos em produção)

1. **COMECE AQUI:** `01_introducao_fairness.ipynb`
   - Por que fairness importa
   - Casos reais de bias (Amazon, COMPAS, Apple Card)
   - Atributos protegidos (gender, race, age)
   - 15 métricas de fairness
   - Primeiro teste de fairness

2. **CRÍTICO:** `02_analise_completa_fairness.ipynb` ⭐⭐
   - **NOTEBOOK MAIS IMPORTANTE DA PASTA**
   - Análise completa com 15 métricas
   - EEOC 80% Rule compliance
   - Análise por grupo (gender, race, age)
   - Confusion matrices por grupo
   - Threshold analysis
   - Relatório HTML profissional
   - Decisão de deploy com checklist legal

3. **Se Detectou Bias:** `03_mitigacao_bias.ipynb`
   - Técnicas Pre/In/Post-processing
   - Reweighting
   - Threshold optimization
   - Comparação Before vs After
   - Trade-offs (accuracy vs fairness)

---

## 📖 O que Você Vai Aprender

### Notebook 1: Introdução a Fairness
- ✅ **Casos Reais de Bias**
  - Amazon (recrutamento)
  - COMPAS (justiça criminal)
  - Apple Card (crédito)
  - Reconhecimento facial
- ✅ **Atributos Protegidos**
  - Gender, race, age, religion, etc.
  - Regulações (EEOC, GDPR, LGPD)
- ✅ **15 Métricas de Fairness**
  - Demographic Parity
  - Equal Opportunity
  - Equalized Odds
  - Disparate Impact (⭐ EEOC)
  - ... e mais 11
- ✅ **Auto-detecção** de atributos sensíveis

### Notebook 2: Análise Completa ⭐⭐ (CRÍTICO)
- ✅ **Cenário Real**: Credit Scoring
- ✅ **15 Métricas Calculadas**
- ✅ **EEOC 80% Rule** - Compliance legal
- ✅ **Análise por Grupo**
  - Gender (Male vs Female)
  - Race (White, Black, Hispanic, Asian, Other)
  - Age groups
- ✅ **Confusion Matrices por Grupo**
- ✅ **Threshold Analysis** - Otimizar fairness
- ✅ **Relatório HTML** para auditoria
- ✅ **Checklist de Deploy** - Decisão legal

### Notebook 3: Mitigação de Bias
- ✅ **3 Tipos de Mitigação**
  - Pre-processing (dados)
  - In-processing (algoritmo)
  - Post-processing (predições)
- ✅ **Técnicas Práticas**
  - Reweighting
  - Threshold optimization
  - Fairness constraints
- ✅ **Comparação Before vs After**
- ✅ **Trade-offs** (accuracy vs fairness)
- ✅ **Bibliotecas Avançadas** (Fairlearn, AIF360)

---

## 🎓 Pré-requisitos

### Conhecimento
- Completar `01_introducao/` (recomendado)
- Entender métricas de classificação (TPR, FPR, Precision, Recall)
- **Importante**: Noções básicas de ética em ML

### Instalação
```bash
pip install deepbridge jupyter pandas numpy matplotlib seaborn scikit-learn

# Opcional (para técnicas avançadas)
pip install fairlearn aif360
```

---

## 🚀 Como Executar

```bash
# 1. Navegar até a pasta
cd /home/guhaase/projetos/DeepBridge/examples/notebooks/04_fairness

# 2. Iniciar Jupyter
jupyter notebook

# 3. Abrir o primeiro notebook
# 01_introducao_fairness.ipynb
```

---

## 💡 Principais Conceitos

### EEOC 80% Rule (Four-Fifths Rule) ⭐

```python
# Regra fundamental de compliance
Disparate Impact = P(Ŷ=1 | Unprivileged) / P(Ŷ=1 | Privileged)

# ✅ PASSA: DI >= 0.80
# ❌ FALHA: DI < 0.80

# Exemplo
male_approval_rate = 0.50  # 50%
female_approval_rate = 0.35  # 35%
di = female_approval_rate / male_approval_rate  # 0.70
# Resultado: ❌ FALHA (< 0.80)
```

### Executar Análise de Fairness

```python
from deepbridge import DBDataset, Experiment

# Criar Experiment COM protected_attributes
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    protected_attributes=['gender', 'race'],  # ← CRÍTICO!
    random_state=42
)

# Executar análise completa
fairness_result = exp.run_fairness_tests(config='full')

# Verificar EEOC compliance
passes_eeoc = fairness_result.passes_eeoc_compliance()
print(f"EEOC Compliance: {'✅ PASSA' if passes_eeoc else '❌ FALHA'}")

# Gerar relatório HTML
exp.save_fairness_report('fairness_report.html')
```

---

## 📊 As 15 Métricas de Fairness

| # | Métrica | O que Mede | Importância |
|---|---------|------------|-------------|
| 1 | Demographic Parity Difference | Taxa de predições positivas igual | Alta |
| 2 | Demographic Parity Ratio | Ratio de taxas de predições | Alta |
| 3 | Equal Opportunity Difference | TPR igual entre grupos | Alta |
| 4 | Equalized Odds Difference | TPR e FPR iguais | Alta |
| 5 | **Disparate Impact** ⭐ | **EEOC 80% Rule** | **CRÍTICA** |
| 6 | Statistical Parity Difference | Similar a Demographic Parity | Média |
| 7 | Average Odds Difference | Média de TPR e FPR | Média |
| 8 | Theil Index | Desigualdade geral | Média |
| 9 | False Positive Rate Difference | FPR diferença | Alta |
| 10 | False Negative Rate Difference | FNR diferença | Alta |
| 11 | Precision Difference | Precisão diferença | Média |
| 12 | Recall Difference | Recall diferença | Média |
| 13 | F1 Score Difference | F1 diferença | Média |
| 14 | Accuracy Difference | Accuracy diferença | Baixa |
| 15 | Selection Rate | Taxa de seleção | Média |

---

## ⚖️ Regulações e Compliance

### 🇺🇸 Estados Unidos
- **EEOC** (Equal Employment Opportunity Commission)
- **Fair Lending Laws**
- **Equal Credit Opportunity Act (ECOA)**
- **Fair Housing Act**
- **Civil Rights Act of 1964**

**Penalidades**: Multas de milhões a bilhões de dólares

### 🇪🇺 União Europeia
- **GDPR** (General Data Protection Regulation)
- **EU AI Act** (proposto)
- **Right to Explanation**

**Penalidades**: Até 4% da receita global anual

### 🇧🇷 Brasil
- **LGPD** (Lei Geral de Proteção de Dados)
- Artigos sobre decisões automatizadas

**Penalidades**: Até R$ 50 milhões por infração

---

## 🚨 Casos Reais de Consequências

### Multas e Settlements (EUA)
- **Bank of America** (2011): **$335 milhões**
- **Wells Fargo** (2012): **$175 milhões**
- **Countrywide Financial** (2011): **$335 milhões**
- **Multiple banks** (2010s): **Bilhões em total**

### Danos à Reputação
- **Amazon** (2018): Sistema de recrutamento descontinuado
- **COMPAS** (2016): Investigações e processos
- **Apple Card** (2019): Investigação regulatória
- **Reconhecimento Facial**: Moratórias e banimentos

---

## 🎯 Decisão: Qual Notebook Usar?

| Sua Situação | Notebook Recomendado |
|--------------|---------------------|
| Nunca analisou fairness antes | `01_introducao_fairness` |
| Modelo vai para produção (critical!) | `02_analise_completa` ⭐⭐ |
| Detectou bias e precisa corrigir | `03_mitigacao_bias` |
| Aplicação regulada (credit, hiring, healthcare) | **TODOS (obrigatório)** |
| Auditoria ou compliance | `02_analise_completa` + relatórios |

---

## ✅ Checklist de Fairness para Produção

Antes de fazer deploy de QUALQUER modelo que afeta pessoas:

- [ ] ✅ Identificar atributos protegidos
- [ ] ✅ Executar análise completa (15 métricas)
- [ ] ✅ Verificar EEOC 80% Rule
- [ ] ✅ Analisar confusion matrices por grupo
- [ ] ✅ Gerar relatório HTML para documentação
- [ ] ✅ Consultar time jurídico
- [ ] ✅ Se bias detectado: aplicar mitigação
- [ ] ✅ Re-validar após mitigação
- [ ] ✅ Configurar monitoramento contínuo em produção
- [ ] ✅ Estabelecer frequência de re-validação

**Se QUALQUER item falhar: NÃO FAZER DEPLOY!**

---

## 🎯 Próximos Passos

Depois de dominar fairness, continue para:

📁 **05_casos_uso/01_credit_scoring.ipynb** ⭐⭐⭐
- Caso real completo end-to-end
- Credit Scoring com compliance total
- Workflow completo de validação

📁 **03_testes_validacao/**
- Combinar fairness com robustez
- Combinar fairness com incerteza

---

## 💡 Dicas Críticas

### 1. Fairness é OBRIGATÓRIO, não opcional
```python
# ❌ NUNCA faça isso em produção:
model.fit(X, y)
# Deploy sem análise de fairness

# ✅ SEMPRE faça isso:
model.fit(X, y)
exp = Experiment(dataset, protected_attributes=['gender', 'race'])
fairness_result = exp.run_fairness_tests(config='full')
if fairness_result.passes_eeoc_compliance():
    # OK para considerar deploy
else:
    # STOP - aplicar mitigação
```

### 2. Use config='full' para Produção
- `quick`: Exploração inicial
- `medium`: Desenvolvimento
- `full`: **Validação final obrigatória**

### 3. Sempre Salve Relatórios
```python
# Documentação para auditoria e compliance
exp.save_fairness_report('fairness_report_YYYY-MM-DD.html')
```

### 4. Monitoramento Contínuo
Fairness pode degradar ao longo do tempo!
- Re-validar mensalmente (mínimo)
- Monitorar em produção
- Alertas automáticos se DI < 0.80

---

## 📚 Recursos Adicionais

### Bibliotecas de Fairness
- **Fairlearn** (Microsoft): https://fairlearn.org
- **AIF360** (IBM): https://aif360.mybluemix.net
- **Themis-ml**: https://github.com/cosmicBboy/themis-ml

### Leituras Recomendadas
- ProPublica - "Machine Bias" (COMPAS)
- Kate Crawford - "Atlas of AI"
- Cathy O'Neil - "Weapons of Math Destruction"
- Solon Barocas et al. - "Fairness and Machine Learning"

### Cursos
- Google - "Machine Learning Fairness"
- Coursera - "AI For Everyone" (Andrew Ng)

---

## 📞 Precisa de Ajuda?

- 📖 [Documentação DeepBridge](../../planejamento_doc/1-CORE/)
- 💻 [Código Fonte](https://github.com/DeepBridge-Validation/DeepBridge)
- ❓ [Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)

**Para questões legais**: Sempre consulte advogados especializados em compliance de ML

---

<div style="background-color: #ffebee; padding: 20px; border-radius: 10px;">
<h3 style="color: #c62828;">⚠️ AVISO LEGAL</h3>
<p style="color: #b71c1c;">
Estes notebooks são educacionais. Para aplicações reais em produção que afetam pessoas:
</p>
<ul style="color: #b71c1c;">
<li><b>SEMPRE</b> consulte time jurídico</li>
<li><b>SEMPRE</b> contrate especialistas em fairness ML</li>
<li><b>SEMPRE</b> siga todas as regulações aplicáveis</li>
<li><b>SEMPRE</b> documente TUDO</li>
<li><b>SEMPRE</b> monitore continuamente</li>
</ul>
<p style="color: #b71c1c;">
<b>O uso inadequado de modelos com bias pode resultar em:</b><br>
- Multas milionárias<br>
- Processos judiciais<br>
- Danos irreversíveis à reputação<br>
- Perda de licença operacional<br>
- Discriminação ilegal de pessoas
</p>
</div>

---

**Última Atualização**: 04 de Novembro de 2025
**Status**: ✅ Fase 2 Completa (3/3 notebooks)
**Importância**: ⭐⭐ CRÍTICA para modelos em produção
