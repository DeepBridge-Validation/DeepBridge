# Casos de Uso: DeepBridge + LangChain

**Documento Complementar:** Exemplos práticos de uso
**Versão:** 1.0
**Data:** Dezembro 2025

---

## 📋 Estrutura deste Documento

1. Casos de Uso por Persona
2. Cenários de Validação Regulatória
3. Cenários de Stress Testing
4. Cenários de Explainability
5. Workflows Complexos
6. Integração com Sistemas Existentes

---

## 1. Casos de Uso por Persona

### 1.1 Data Scientist

**Perfil:** Desenvolve modelos ML, precisa validá-los rapidamente

**Caso de Uso: Validação Rápida Pré-Deployment**

```python
from deepbridge import DBDataset, ValidationAgent
from langchain.chat_models import ChatOpenAI

# Setup
dataset = DBDataset(
    data=df_test,
    target_column='churn',
    model=my_xgboost_model
)

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = ValidationAgent(dataset=dataset, llm=llm)

# Validação em linguagem natural
result = agent.run("""
Preciso validar este modelo de churn antes do deploy.

Prioridades:
1. Robustez - modelo vai para produção, precisa ser estável
2. Não posso ter falsos positivos altos (precisão importante)
3. Incerteza - quero saber quando o modelo está incerto

Execute testes nível médio e me diga se posso deployar.
""")

# Output
print(result['answer'])
# """
# VALIDAÇÃO PRÉ-DEPLOYMENT: Modelo de Churn
#
# Executei 3 testes (robustez, métricas de performance, incerteza):
#
# ✓ ROBUSTEZ: APROVADO
# - Score: 0.84 (bom)
# - Degradação média: 9% sob perturbações
# - Weak spot: feature 'last_login_days' (15% degradação)
#
# ⚠️ PERFORMANCE: ATENÇÃO
# - Precisão: 0.72 (abaixo de ideal 0.80 para baixos FP)
# - Recall: 0.85 (bom)
# - F1: 0.78
#
# ✓ INCERTEZA: BOA CALIBRAÇÃO
# - Coverage: 94.8% (target: 95%)
# - Calibration error: 0.04 (bem calibrado)
#
# DECISÃO DE DEPLOY: CONDICIONAL
# - Modelo é robusto e bem calibrado
# - PORÉM: precisão pode gerar FP altos
#
# RECOMENDAÇÕES:
# 1. [CRÍTICO] Ajustar threshold para aumentar precisão (atual: 0.5 → 0.6)
# 2. [ALTO] Revisar feature 'last_login_days' (weak spot)
# 3. [MÉDIO] Adicionar regra de negócio para casos de baixa confiança
#
# APROVAÇÃO PARA DEPLOY: SIM, com ajustes de threshold
# """

# Ajustar threshold conforme recomendado
# ... código de ajuste ...
```

**Tempo economizado:** 4h de análise manual → 5 minutos
**Valor:** Feedback imediato e acionável

---

### 1.2 ML Engineer

**Perfil:** Gerencia modelos em produção, precisa monitorar drift

**Caso de Uso: Monitoramento Contínuo de Modelos**

```python
# Setup para modelo em produção
production_dataset = DBDataset(
    data=production_data_last_week,
    target_column='default',
    model=load_model('production_credit_model_v2.3.pkl')
)

agent = ValidationAgent(dataset=production_dataset, llm=llm)

# Monitoramento semanal automatizado
result = agent.run("""
Este é o modelo de crédito em produção (v2.3).

Compare performance com baseline (semana passada):
- Baseline robustness: 0.91
- Baseline fairness: EEOC compliant
- Baseline AUC: 0.84

Identifique qualquer degradação e sinalize urgência.
""")

print(result['answer'])
# """
# MONITORAMENTO SEMANAL: Credit Model v2.3
#
# ❌ ALERTA: DEGRADAÇÃO DETECTADA
#
# ROBUSTEZ:
# - Score atual: 0.84 (baseline: 0.91) → DEGRADAÇÃO DE 7.7%
# - Status: WARNING
# - Causa provável: Distribuição de 'income' shifted
#
# FAIRNESS:
# - EEOC: AINDA COMPLIANT (impact ratio: 0.82)
# - Demographic parity: 0.09 (vs. 0.06 baseline) → PIOROU
#
# PERFORMANCE:
# - AUC: 0.82 (vs. 0.84 baseline) → LEVE DEGRADAÇÃO
#
# ANÁLISE DE URGÊNCIA: ALTA
# - Robustez caiu abaixo de threshold aceitável (0.85)
# - Tendência de degradação em 3 semanas consecutivas
#
# AÇÕES RECOMENDADAS:
# 1. [URGENTE] Investigar data drift em 'income' feature
# 2. [URGENTE] Considerar retreinamento com dados recentes
# 3. [ALTO] Aumentar frequência de monitoramento (semanal → diário)
# 4. [MÉDIO] Revisar pipeline de features upstream
#
# PRÓXIMOS PASSOS:
# - Executar análise detalhada de drift de dados
# - Agendar reunião com equipe de dados
# - Preparar modelo v2.4 com dados recentes
# """

# Trigger pipeline de retreinamento
if result['metadata']['urgency'] == 'HIGH':
    trigger_retraining_pipeline(model_version='v2.4')
```

**Tempo economizado:** 8h/semana de monitoramento manual → 10 min automatizado
**Valor:** Detecção proativa de problemas

---

### 1.3 Compliance Officer

**Perfil:** Responsável por compliance regulatório, não é técnico

**Caso de Uso: Auditoria para Reguladores**

```python
# Modelo de hiring - precisa compliance EEOC
hiring_dataset = DBDataset(
    data=df_applicants_2024,
    target_column='hired',
    model=hiring_ai_model,
    categorical_features=['gender', 'race', 'age_group']
)

agent = ValidationAgent(
    dataset=hiring_dataset,
    llm=llm,
    protected_attributes=['gender', 'race', 'age_group']
)

# Auditoria regulatória
result = agent.run("""
Preciso preparar relatório de compliance EEOC para auditoria regulatória.

O auditor vai verificar:
1. Se respeitamos four-fifths rule (80%)
2. Se há disparate impact estatisticamente significativo
3. Justificativas de business necessity se houver violações
4. Evidência de mitigações implementadas

Gere análise completa formatada para regulador não-técnico.
Foque em linguagem clara e conforme à regulação.
""")

print(result['answer'])
# """
# RELATÓRIO DE COMPLIANCE EEOC
# Hiring AI Model - Ano 2024
# Prepared for: Regulatory Audit
#
# ===== SUMÁRIO EXECUTIVO =====
#
# STATUS GERAL: ⚠️ PARCIALMENTE COMPLIANT
# - Four-Fifths Rule: 2 de 3 grupos PASS
# - Disparate Impact: 1 violação detectada
# - Mitigações: Implementadas (parcialmente efetivas)
#
# ===== ANÁLISE DETALHADA =====
#
# 1. FOUR-FIFTHS RULE (29 CFR § 1607.4D)
#
#    Grupo: GENDER
#    - Female selection rate: 42.3%
#    - Male selection rate: 45.1%
#    - Impact Ratio: 0.938 (93.8%)
#    - Status: ✓ PASS (≥ 80%)
#
#    Grupo: RACE (Black vs. White)
#    - Black selection rate: 31.2%
#    - White selection rate: 44.8%
#    - Impact Ratio: 0.696 (69.6%)
#    - Status: ✗ FAIL (< 80%)
#    - Statistical Significance: p=0.003 (SIGNIFICANT)
#
#    Grupo: AGE (Under 40 vs. Over 40)
#    - Under 40 selection rate: 43.7%
#    - Over 40 selection rate: 42.9%
#    - Impact Ratio: 1.019 (101.9%)
#    - Status: ✓ PASS (≥ 80%)
#
# 2. BUSINESS NECESSITY ANALYSIS
#
#    Para violação em RACE:
#    - Features com maior correlação com raça:
#      1. 'years_of_experience' (r=0.31)
#      2. 'education_level' (r=0.24)
#      3. 'previous_industry' (r=0.18)
#
#    - Justificativa de Business Necessity:
#      "Years of experience é preditor válido de performance
#       (correlação com job performance: 0.67). Porém, correlação
#       com raça sugere proxy effect."
#
#    - Status: BUSINESS NECESSITY NÃO TOTALMENTE COMPROVADA
#
# 3. MITIGAÇÕES IMPLEMENTADAS
#
#    ✓ Threshold adjustment: Implementado
#      - Impact ratio melhorou de 0.64 → 0.696
#      - Ainda abaixo de 80%, mas em direção correta
#
#    ✓ Feature engineering: Em andamento
#      - Removida feature 'zip_code' (proxy para raça)
#      - Adicionada normalização de 'years_of_experience'
#
#    ⏳ Fairness constraints no treinamento: Planejado
#      - Estimativa de implementação: Q1 2026
#
# 4. RECOMENDAÇÕES PARA COMPLIANCE TOTAL
#
#    [CRÍTICO - 30 dias]
#    - Implementar fairness constraints (demographic parity)
#    - Target: Impact ratio ≥ 0.82
#
#    [ALTO - 60 dias]
#    - Re-validar business necessity com estudo empírico
#    - Documentar relação features ↔ job performance
#
#    [MÉDIO - 90 dias]
#    - Implementar monitoramento contínuo de fairness
#    - Criar dashboard para compliance tracking
#
# 5. EVIDÊNCIAS PARA AUDITOR
#
#    Anexos incluídos:
#    - Confusion matrices por grupo demográfico
#    - Análise estatística completa (chi-square, Fisher's exact)
#    - Historical trend de impact ratios (12 meses)
#    - Documentação de mitigações implementadas
#
# ===== CERTIFICAÇÃO =====
#
# Este relatório foi gerado usando DeepBridge v2.0,
# framework validado para compliance EEOC.
#
# Métricas calculadas de forma determinística e auditável.
# Logs de execução disponíveis para inspeção regulatória.
#
# Data: 2025-12-06
# Audit Trail ID: session_a8f3c2d91b45
# """

# Salvar para regulador
with open('eeoc_compliance_report_2024.txt', 'w') as f:
    f.write(result['answer'])

# Salvar audit trail
agent.save_audit_trail('eeoc_audit_trail_2024.json')
```

**Tempo economizado:** 40h de auditoria manual → 2h com agente
**Valor:** Relatório formatado e conforme, evidências rastreáveis

---

## 2. Cenários de Validação Regulatória

### 2.1 SR 11-7 (Model Risk Management - Bancos)

**Contexto:** Banco precisa validar modelo de risco de crédito para compliance SR 11-7

```python
# Modelo de PD (Probability of Default)
from deepbridge import ValidationAgent, StressTestAgent

pd_dataset = DBDataset(
    data=loan_portfolio_data,
    target_column='default_12m',
    model=pd_model_v3,
    features=economic_features + borrower_features
)

# 1. Validação Conceitual (SR 11-7 §5)
validation_agent = ValidationAgent(dataset=pd_dataset, llm=llm)

conceptual_validation = validation_agent.run("""
Validação conceitual conforme SR 11-7 para modelo de PD (Probability of Default).

Verifique:
1. Robustez do modelo a variações econômicas
2. Calibração de probabilidades (essencial para PD)
3. Discriminação (AUC, Gini)
4. Stability across time periods

Gere relatório técnico para Model Risk Management committee.
""")

# 2. Backtesting (SR 11-7 §6)
backtest_result = validation_agent.run("""
Execute backtesting do modelo de PD nos últimos 24 meses.

Compare:
- PD prevista vs. default rate realizado
- Binning analysis (10 bins por PD)
- Hosmer-Lemeshow test

Identifique períodos de underprediction/overprediction.
""")

# 3. Stress Testing (SR 11-7 §7)
stress_agent = StressTestAgent(dataset=pd_dataset, llm=llm)

stress_result = stress_agent.run("""
Simule cenários de stress conforme supervisory scenarios do Fed:

Cenário Severely Adverse:
- GDP: -4.5%
- Unemployment: +6%
- House prices: -20%
- Corporate spreads: +300bps

Avalie:
1. Aumento projetado na PD média
2. Migration de ratings
3. Quebra de correlações históricas
4. Model stability under stress
""")

# 4. Sensibilidade (SR 11-7 §8)
sensitivity_result = validation_agent.run("""
Análise de sensibilidade do modelo de PD.

Identifique:
- Top 5 features mais influentes
- Limites de aplicabilidade do modelo
- Scenarios onde modelo degrada
- Recommended monitoring metrics
""")

# Compilar relatório SR 11-7
sr117_report = generate_sr117_report(
    conceptual=conceptual_validation,
    backtest=backtest_result,
    stress=stress_result,
    sensitivity=sensitivity_result
)

# Output para MRM Committee
save_report(sr117_report, 'SR_11-7_Validation_Report_PD_Model_v3.pdf')
```

**Conformidade:**
- ✅ Validação conceitual (§5)
- ✅ Ongoing monitoring (§6)
- ✅ Stress testing (§7)
- ✅ Sensitivity analysis (§8)
- ✅ Documentation (§9)

---

### 2.2 EU AI Act (High-Risk AI Systems)

**Contexto:** Sistema de scoring de crédito classificado como "high-risk" sob EU AI Act

```python
# AI Act requer: transparency, human oversight, accuracy, robustness

ai_act_agent = ValidationAgent(dataset=credit_dataset, llm=llm)

result = ai_act_agent.run("""
Validação conforme EU AI Act (Artigos 9-15) para sistema de crédito.

Requisitos a verificar:

Art. 9 - Risk Management:
- Identificar riscos residuais após mitigações
- Documentar medidas de risk mitigation

Art. 10 - Data Governance:
- Verificar qualidade e representatividade dos dados
- Detectar viés nos dados de treinamento

Art. 13 - Transparency:
- Gerar explicações compreensíveis para decisões
- Documentar lógica do sistema

Art. 14 - Human Oversight:
- Identificar casos que requerem revisão humana
- Definir thresholds para escalation

Art. 15 - Accuracy, Robustness, Cybersecurity:
- Medir accuracy em produção
- Testar robustez a adversarial examples
- Avaliar resilience a data poisoning

Gere relatório de conformidade técnica.
""")

print(result['answer'])
# """
# EU AI ACT COMPLIANCE REPORT
# High-Risk AI System: Credit Scoring
#
# ===== COMPLIANCE STATUS =====
#
# Art. 9 (Risk Management): ✓ COMPLIANT
# Art. 10 (Data Governance): ⚠️ PARTIALLY COMPLIANT
# Art. 13 (Transparency): ✓ COMPLIANT
# Art. 14 (Human Oversight): ✓ COMPLIANT
# Art. 15 (Accuracy/Robustness): ⚠️ REQUIRES ATTENTION
#
# ===== DETAILED FINDINGS =====
#
# [Art. 9] RISK MANAGEMENT
#
# Riscos Identificados:
# 1. Discrimination risk: MEDIUM (mitigation: fairness constraints)
# 2. Performance degradation: LOW (mitigation: monitoring)
# 3. Adversarial manipulation: MEDIUM (mitigation: input validation)
#
# Residual Risks:
# - Distributional shift em crises econômicas (LOW)
# - Proxy discrimination via correlações (MEDIUM)
#
# Status: ✓ Risk management plan adequado
#
# [Art. 10] DATA GOVERNANCE
#
# Training Data Quality:
# - Representatividade: 89% (target: >90%)
# - Completeness: 97%
# - Consistency: 94%
#
# ⚠️ ISSUE: Underrepresentation de idade 18-25 (6% vs. 12% população)
#
# Bias Detection:
# - Gender bias: NONE DETECTED
# - Age bias: DETECTED (young applicants underrepresented)
# - Geographic bias: MINOR (rural areas -3%)
#
# Status: ⚠️ Requires data augmentation for young applicants
#
# [Art. 13] TRANSPARENCY
#
# Explainability:
# - Global explanations: ✓ Available (feature importance)
# - Local explanations: ✓ Available (SHAP values)
# - Counterfactuals: ✓ Implemented
#
# User Communication:
# - Rejection letters include top 3 reasons
# - Plain language explanations tested with users
#
# Status: ✓ Transparency requirements met
#
# [Art. 14] HUMAN OVERSIGHT
#
# Human-in-the-loop triggers:
# - Low confidence predictions (prob < 0.6 or > 0.4): 18% de casos
# - Borderline decisions (score 0.45-0.55): 12% de casos
# - Adverse decisions for protected groups: 100% de casos
#
# Override capability:
# - Humans can override 100% of decisions
# - Override rate atual: 3.2%
#
# Status: ✓ Human oversight properly implemented
#
# [Art. 15] ACCURACY, ROBUSTNESS, CYBERSECURITY
#
# Accuracy (Production):
# - AUC: 0.81 (target: >0.80) ✓
# - Precision: 0.74 (target: >0.70) ✓
# - Recall: 0.79 (target: >0.75) ✓
#
# Robustness:
# - Robustness score: 0.77 (target: >0.80) ⚠️
# - Adversarial robustness: NOT TESTED ❌
#
# ⚠️ ISSUE: Robustness score abaixo de target
# ❌ CRITICAL: Adversarial robustness não avaliada
#
# Status: ⚠️ Requires robustness improvements
#
# ===== ACTION PLAN =====
#
# CRITICAL (30 dias):
# 1. Implementar testes de adversarial robustness
# 2. Augment training data com idade 18-25
#
# HIGH (60 dias):
# 3. Melhorar robustness score para >0.80
# 4. Documentar todas mitigações formalmente
#
# MEDIUM (90 dias):
# 5. Implementar continuous monitoring dashboard
# 6. Treinar equipe em human oversight procedures
#
# ===== CERTIFICATION =====
#
# Overall Compliance: CONDITIONAL
# - System can operate with enhanced monitoring
# - Critical issues must be resolved within 30 days
# - Re-assessment required after mitigations
# """
```

---

## 3. Stress Testing Avançado

### 3.1 Cenário Econômico: Recessão Severa

```python
from deepbridge import StressTestAgent

stress_agent = StressTestAgent(dataset=lending_dataset, llm=llm)

result = stress_agent.run("""
Simule recessão severa similar a 2008-2009:

Macro shocks:
- GDP: -5% (vs. baseline +2%)
- Unemployment: +8% (3% → 11%)
- Housing prices: -25%
- Stock market: -40%
- Corporate defaults: +250%

Micro impacts:
- Household income: -15% (demissões + salary cuts)
- Debt service ratios: +30% (fixed debts, lower income)
- Savings rate: -50% (depleção de emergências)

Modelo: Credit Default Prediction

Avalie:
1. Aumento projetado em default rate
2. Performance degradation do modelo
3. Calibration breakdown (se probabilidades permanecem válidas)
4. Feature importance shifts
5. Recommendations para model recalibration
""")

print(result['answer'])
# """
# STRESS TEST: Recessão Severa (2008-style)
#
# ===== IMPACTO PROJETADO =====
#
# Default Rate:
# - Baseline: 4.2%
# - Stressed: 12.8%
# - Aumento: +8.6pp (+205%)
#
# ===== MODEL PERFORMANCE =====
#
# Discriminação (AUC):
# - Baseline: 0.82
# - Stressed: 0.74
# - Degradação: -9.8% ⚠️
#
# Calibração:
# - Baseline: Well calibrated (ECE=0.04)
# - Stressed: POOR calibration (ECE=0.18) ❌
# - Observação: Modelo UNDERPREDICTS defaults em stress
#
# Robustness Score:
# - Baseline: 0.85
# - Stressed: 0.61
# - Degradação: -28% 🔴 CRITICAL
#
# ===== FEATURE IMPORTANCE SHIFTS =====
#
# Baseline Top 3:
# 1. Credit score (importance: 0.31)
# 2. DTI ratio (importance: 0.24)
# 3. Income (importance: 0.19)
#
# Stressed Top 3:
# 1. Employment status (importance: 0.38) ↑
# 2. Savings/Emergency fund (importance: 0.29) ↑↑
# 3. DTI ratio (importance: 0.22)
#
# → Shift significativo: Emprego e savings tornam-se críticos
#
# ===== CALIBRATION BREAKDOWN =====
#
# Bins de Probabilidade vs. Default Real:
#
# Prob Bin | Baseline | Stressed | Gap
# ---------|----------|----------|--------
# 0-10%    |  2.1%    | 5.8%     | +3.7pp ⚠️
# 10-20%   |  12.3%   | 18.1%    | +5.8pp ⚠️
# 20-30%   |  23.7%   | 34.2%    | +10.5pp 🔴
# 30-40%   |  34.1%   | 48.9%    | +14.8pp 🔴
# ...
#
# → Modelo subestima risco sistematicamente em stress
#
# ===== RECOMENDAÇÕES =====
#
# [CRÍTICO] RECALIBRAÇÃO OBRIGATÓRIA
# 1. Re-treinar modelo com dados de crise (2008-2009, 2020)
# 2. Adicionar features macroeconômicas:
#    - Unemployment rate
#    - GDP growth
#    - Consumer confidence index
# 3. Implementar dynamic calibration ajustada ao ciclo econômico
#
# [ALTO] AJUSTES DE RISCO
# 4. Aumentar provisões de risco em 150-200%
# 5. Tighten credit standards:
#    - Minimum savings: 6 months → 9 months
#    - Maximum DTI: 40% → 35%
# 6. Enhanced monitoring de employment sectors vulneráveis
#
# [MÉDIO] INFRAESTRUTURA
# 7. Implementar stress testing mensal (vs. trimestral atual)
# 8. Create early warning system baseado em leading indicators
# 9. Develop contingency model para cenários de crise
#
# ===== TIMELINE DE AÇÕES =====
#
# Imediato (hoje):
# - Suspend auto-approval para applications com:
#   * DTI > 35%
#   * Savings < 6 months
#   * Employment em setores vulneráveis (hospitality, retail)
#
# 30 dias:
# - Deploy modelo recalibrado com dados de crise
# - Implementar dynamic provisioning
#
# 60 dias:
# - Launch early warning dashboard
# - Train staff em decisões manuais para casos borderline
#
# 90 dias:
# - Complete stress testing framework overhaul
# - Validate contingency model
# """
```

---

## 4. Explainability para Clientes

### 4.1 Geração de Adverse Action Notices (ECOA)

```python
from deepbridge import ExplainabilityAgent

explain_agent = ExplainabilityAgent(dataset=lending_dataset, llm=llm)

# Cliente específico recusado
customer_id = '12345'
customer_data = df[df['customer_id'] == customer_id]

result = explain_agent.run(f"""
Cliente ID {customer_id} foi RECUSADO para empréstimo pessoal.

Gere Adverse Action Notice conforme ECOA (Equal Credit Opportunity Act):

Requisitos ECOA:
1. Notificação dentro de 30 dias
2. Top 4 razões ESPECÍFICAS (não genéricas)
3. Verificar se razões envolvem atributos protegidos (PROIBIDO)
4. Fornecer nome e endereço do credor
5. Informar direito de solicitar explicação adicional
6. Statement de direitos sob ECOA

Dados do cliente:
{customer_data.to_dict('records')[0]}

Dados agregados para comparação:
- Taxa de aprovação geral: 76%
- Score médio de aprovados: 720
- DTI médio de aprovados: 32%

Gere carta formatada para envio ao cliente.
Tom: profissional mas empático.
""")

print(result['answer'])
# """
# [Letterhead: Acme Lending Inc.]
#
# December 6, 2025
#
# John Smith
# 123 Main Street
# Anytown, ST 12345
#
# Re: Credit Application #APP-2024-12345
#     Adverse Action Notice
#
# Dear Mr. Smith,
#
# Thank you for your recent application for a personal loan with
# Acme Lending Inc. After careful review, we regret to inform you
# that we are unable to approve your application at this time.
#
# This decision was based on information in your credit application
# and your credit report from Experian. The principal reasons for
# our decision are:
#
# PRIMARY REASONS FOR CREDIT DENIAL:
#
# 1. Debt-to-Income Ratio Too High
#    Your ratio: 48%
#    Our maximum: 40%
#    This means your monthly debt payments ($2,880) are too high
#    relative to your gross monthly income ($6,000).
#
# 2. Insufficient Credit History
#    Credit history length: 14 months
#    Preferred minimum: 24 months
#    We require a longer track record of credit management.
#
# 3. Recent Late Payment on Credit Card
#    Late payment reported: August 2025 (30 days past due)
#    This indicates potential difficulty in meeting payment obligations.
#
# 4. Limited Liquid Savings
#    Verified savings: $1,200
#    Recommended minimum: $3,500 (6 months of loan payments)
#    Insufficient reserves for unexpected financial challenges.
#
# FAIRNESS VERIFICATION:
# ✓ No protected class factors (race, gender, religion, national origin,
#   marital status, age) were considered in this decision.
# ✓ This decision was based solely on creditworthiness factors.
# ✓ The same standards apply to all applicants.
#
# CREDIT REPORT INFORMATION:
# The credit report was obtained from:
#   Experian
#   P.O. Box 9701
#   Allen, TX 75013
#   www.experian.com
#   1-888-397-3742
#
# You have the right to obtain a free copy of your credit report from
# this credit bureau within 60 days. You also have the right to dispute
# the accuracy or completeness of any information in your credit report.
#
# NEXT STEPS:
#
# We encourage you to take the following steps to improve your
# creditworthiness for future applications:
#
# 1. Reduce your debt-to-income ratio by:
#    - Paying down existing debts
#    - Increasing your income
#    Target DTI: Below 40% (currently 48%)
#
# 2. Build your credit history:
#    - Continue making on-time payments
#    - Keep credit card balances low
#    - Avoid new credit inquiries
#    Wait until: At least 24 months of history
#
# 3. Build emergency savings:
#    - Target: 6-9 months of expenses
#    - This demonstrates financial resilience
#
# 4. Maintain payment timeliness:
#    - Set up automatic payments
#    - Create payment reminders
#
# You may reapply once you have addressed these factors. We estimate
# you could be eligible in approximately 6-12 months with:
# - DTI reduced to 38% or below
# - 6 months of on-time payments
# - Savings increased to $3,500+
#
# YOUR RIGHTS UNDER ECOA:
#
# The federal Equal Credit Opportunity Act prohibits creditors from
# discriminating against credit applicants on the basis of race, color,
# religion, national origin, sex, marital status, age (provided the
# applicant has the capacity to enter into a binding contract);
# because all or part of the applicant's income derives from any
# public assistance program; or because the applicant has in good
# faith exercised any right under the Consumer Credit Protection Act.
#
# If you have questions about this decision or would like to discuss
# your application, please contact us:
#
#   Acme Lending Inc.
#   Customer Service Department
#   Phone: 1-800-555-0123
#   Email: customerservice@acmelending.com
#   Hours: Monday-Friday, 8am-6pm EST
#
# We appreciate your interest in Acme Lending and wish you success
# in your financial goals.
#
# Sincerely,
#
# Jane Doe
# Credit Operations Manager
# Acme Lending Inc.
#
# ===== INTERNAL NOTES (NOT FOR CUSTOMER) =====
#
# Model Decision:
# - Predicted probability of default: 0.62 (threshold: 0.50)
# - Risk tier: HIGH
# - Automatic decision: DECLINE
#
# Fairness Check:
# - Protected attributes: NOT USED in decision
# - Decision explainable without protected factors: YES
# - Disparate impact check: PASS
#
# Compliance:
# ✓ ECOA notice requirements met
# ✓ Specific reasons provided (not generic)
# ✓ Rights statement included
# ✓ Credit bureau information provided
# ✓ Timeframe: Within 30 days of application
#
# Audit Trail ID: session_explain_12345_20251206
# """
```

**Conformidade ECOA:**
- ✅ Notificação dentro de 30 dias
- ✅ Razões específicas (não genéricas)
- ✅ Sem uso de atributos protegidos
- ✅ Informação de credit bureau
- ✅ Statement de direitos
- ✅ Acionável (cliente sabe como melhorar)

---

## 5. Workflows Complexos

### 5.1 Pipeline Completo: Desenvolvimento → Produção

```python
# ===== FASE 1: DESENVOLVIMENTO =====

# Data Scientist treina modelo
model = train_xgboost_model(X_train, y_train)

# Validação rápida
dev_agent = ValidationAgent(dataset=dev_dataset, llm=llm)
dev_result = dev_agent.run("Validação rápida: robustez e performance básica")

if dev_result['metadata']['approval'] == 'PASS':
    # ===== FASE 2: VALIDAÇÃO PRÉ-DEPLOYMENT =====

    validation_agent = ValidationAgent(dataset=test_dataset, llm=llm)

    full_validation = validation_agent.run("""
    Validação completa pré-deployment (nível FULL):
    1. Robustez
    2. Fairness (EEOC compliance)
    3. Incerteza
    4. Resilience
    5. Hyperparameter importance

    Gere relatório para Model Risk Management.
    """)

    if full_validation['metadata']['mrm_approval'] == 'APPROVED':
        # ===== FASE 3: STRESS TESTING =====

        stress_agent = StressTestAgent(dataset=test_dataset, llm=llm)

        stress_results = stress_agent.run("""
        Stress testing regulatório:
        - Cenário base
        - Cenário adverse
        - Cenário severely adverse

        Avalie model stability.
        """)

        if stress_results['metadata']['stress_approved']:
            # ===== FASE 4: DEPLOYMENT =====

            deploy_model(model, version='v2.0')

            # ===== FASE 5: MONITORAMENTO CONTÍNUO =====

            # Setup monitoramento semanal
            schedule_monitoring(
                model_version='v2.0',
                frequency='weekly',
                agent_config={
                    'tests': ['robustness', 'fairness', 'performance'],
                    'alerts': ['degradation', 'drift', 'violations']
                }
            )

# Monitoramento executado automaticamente
def weekly_monitoring():
    prod_data = get_production_data(last_days=7)
    prod_dataset = DBDataset(data=prod_data, model=load_model('v2.0'))

    monitor_agent = ValidationAgent(dataset=prod_dataset, llm=llm)

    result = monitor_agent.run("""
    Monitoramento semanal - compare com baseline:
    - Robustness score baseline: 0.89
    - Fairness baseline: EEOC compliant
    - AUC baseline: 0.84

    Alerte se degradação > 5% em qualquer métrica.
    """)

    if result['metadata']['alerts']:
        send_alert_to_team(result)
        trigger_investigation(result['metadata']['alerts'])
```

---

*(Continua...)*

**Este documento contém mais 5 seções adicionais. Deseja que eu continue com:**
- 5.2 Comparação A/B de Modelos
- 6. Integração com Sistemas Existentes (MLflow, SageMaker, etc.)
- 7. Templates de Prompts Reutilizáveis
- 8. Best Practices e Anti-Patterns
- 9. Troubleshooting e FAQ

Total estimado: ~15,000 palavras adicionais.
