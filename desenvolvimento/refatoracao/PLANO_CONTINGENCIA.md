# Plano de Contingência - DeepBridge v2.0

**Última atualização:** 2026-02-16

Este documento define procedimentos de contingência para lidar com bugs críticos, falhas de release e outras emergências no ecossistema DeepBridge.

---

## 📋 Índice

1. [Definições e Severidade](#definições-e-severidade)
2. [Equipe de Resposta](#equipe-de-resposta)
3. [Procedimentos de Emergência](#procedimentos-de-emergência)
4. [Rollback de Release](#rollback-de-release)
5. [Comunicação de Crise](#comunicação-de-crise)
6. [Templates de Comunicação](#templates-de-comunicação)
7. [Pós-Mortem](#pós-mortem)

---

## Definições e Severidade

### 🔴 Severidade 1 (S1) - CRÍTICO

**Definição:**
- Sistema completamente inutilizável
- Perda/corrupção de dados
- Vulnerabilidade de segurança ativa
- Afeta >75% dos usuários

**Exemplos:**
- Pacote não pode ser importado (`ModuleNotFoundError`)
- Crash ao inicializar qualquer funcionalidade
- Vulnerabilidade de segurança explorada ativamente
- Dependência quebrada que bloqueia instalação

**SLA:**
- **Tempo de resposta:** < 2 horas
- **Tempo de resolução:** < 24 horas
- **Escalação:** Imediata para todos os maintainers

---

### 🟠 Severidade 2 (S2) - ALTO

**Definição:**
- Funcionalidade principal não funciona
- Workaround existe mas é difícil
- Afeta 25-75% dos usuários

**Exemplos:**
- Função principal retorna resultado incorreto
- Performance degradada significativamente (>10x mais lenta)
- Incompatibilidade com versão comum de Python/dependência

**SLA:**
- **Tempo de resposta:** < 8 horas
- **Tempo de resolução:** < 3 dias
- **Escalação:** Mantainer principal

---

### 🟡 Severidade 3 (S3) - MÉDIO

**Definição:**
- Funcionalidade secundária não funciona
- Workaround simples existe
- Afeta <25% dos usuários

**Exemplos:**
- Parâmetro opcional não funciona
- Documentação incorreta causa confusão
- Warning inesperado mas não prejudicial

**SLA:**
- **Tempo de resposta:** < 24 horas
- **Tempo de resolução:** < 1 semana
- **Escalação:** Não necessária

---

### 🟢 Severidade 4 (S4) - BAIXO

**Definição:**
- Problema cosmético
- Não afeta funcionalidade
- Impacto mínimo

**Exemplos:**
- Typo em mensagem
- Log desnecessário
- Formatação de código

**SLA:**
- **Tempo de resposta:** < 48 horas
- **Tempo de resolução:** Próximo release
- **Escalação:** Não necessária

---

## Equipe de Resposta

### Papéis e Responsabilidades

#### Lead Maintainer (Responsável Principal)
- **Nome:** [Definir]
- **Contato:** [Email, Phone, Discord]
- **Responsabilidades:**
  - Decisão final sobre rollback
  - Coordenação da resposta
  - Comunicação externa principal

#### Technical Lead (Líder Técnico)
- **Nome:** [Definir]
- **Contato:** [Email, Phone, Discord]
- **Responsabilidades:**
  - Análise técnica do problema
  - Implementação de fixes
  - Coordenação com DevOps

#### Community Manager (Gestor de Comunidade)
- **Nome:** [Definir]
- **Contato:** [Email, Phone, Discord]
- **Responsabilidades:**
  - Comunicação com usuários
  - Gerenciamento de issues
  - FAQs e suporte

---

### Canais de Comunicação de Emergência

#### Interno
- **Primary:** Discord #emergencies (ou Slack)
- **Secondary:** Email thread marcado [CRITICAL]
- **Tertiary:** Phone/SMS (para S1)

#### Externo
- **Primary:** GitHub Issues (pinned)
- **Secondary:** Twitter/X (@deepbridge)
- **Tertiary:** Email blast (se mailing list existir)

---

## Procedimentos de Emergência

### Procedimento para S1 (Crítico)

#### 1. Detecção e Alerta (0-15 min)

**Quando detectado:**
```bash
# Criar issue CRÍTICA imediatamente
gh issue create \
  --title "[CRITICAL S1] Brief description" \
  --label "bug,priority:critical,severity:s1" \
  --body "$(cat <<'EOF'
## SEVERITY: S1 - CRITICAL

**Impact:** [Describe impact - e.g., "All users cannot install package"]
**Affected Versions:** [e.g., 2.0.0, 2.0.1]
**Discovered:** [Date/Time]
**Reporter:** [Who found it]

## Immediate Actions
- [ ] Issue created and pinned
- [ ] Team notified
- [ ] Workaround identified (if any)
- [ ] Fix in progress

## Details
[Detailed description, stack trace, etc.]
EOF
)"

# Pin issue
gh issue pin [issue-number]
```

**Notificar equipe:**
```
@everyone CRITICAL S1 INCIDENT

Issue: #[number]
Impact: [Brief description]
ETA for fix: [Estimate or "investigating"]

Action items:
- [Lead] Coordinating response
- [Tech] Investigating root cause
- [Community] Preparing communication

War room: [Discord link]
```

---

#### 2. Avaliação e Decisão (15-30 min)

**Questões a responder:**
1. Qual a extensão do impacto?
2. Existe workaround viável?
3. Podemos fazer hotfix ou precisamos de rollback?
4. Quais versões são afetadas?

**Matriz de Decisão:**

| Situação | Ação |
|----------|------|
| Bug em versão mais recente + versão anterior funciona | **Rollback** + comunicar downgrade |
| Bug em todas as versões + fix rápido possível (<4h) | **Hotfix** imediato |
| Bug em todas as versões + fix complexo (>4h) | **Rollback** + fix planejado |
| Vulnerabilidade de segurança | **Yank** do PyPI + hotfix urgente |

---

#### 3. Execução (30 min - 24h)

**Opção A: Hotfix**
```bash
# Ver WORKFLOW_BUGFIX.md seção "Workflow de Hotfix"

# 1. Branch de hotfix
git checkout -b hotfix/critical-s1-issue-[n]

# 2. Fix mínimo + teste
# [Implementar fix]

# 3. Test
pytest tests/ -v

# 4. Commit e PR
git commit -m "hotfix: critical S1 - [description]

CRITICAL S1: [Impact]

- Fix: [What was fixed]
- Test: [Test added]
- Verification: [How tested]

Fixes #[issue-number]"

# 5. Fast-track review e merge
gh pr create --label "priority:critical,severity:s1"

# 6. Release imediato após merge
```

**Opção B: Rollback (ver seção específica)**

---

#### 4. Comunicação (Paralelo à execução)

**Comunicado inicial (0-30 min):**
```markdown
🚨 CRITICAL ISSUE DETECTED - DeepBridge v2.0.X

We have identified a critical issue affecting [description].

**Impact:** [Who is affected]
**Status:** Investigating
**Workaround:** [If available]

We are working on a fix and will update every hour.

Track: https://github.com/guhaase/DeepBridge/issues/[n]
```

**Updates a cada 1 hora:**
```markdown
UPDATE [HH:MM UTC]: [Status update]

- Current status: [Investigating/Fix in progress/Testing]
- ETA: [Estimate]
- Workaround: [If discovered]
```

**Comunicado de resolução:**
```markdown
✅ RESOLVED - DeepBridge v2.0.X Critical Issue

The critical issue has been resolved in v2.0.Y.

**Action Required:**
pip install --upgrade deepbridge

**Details:** https://github.com/guhaase/DeepBridge/releases/tag/v2.0.Y

Thank you for your patience.
```

---

#### 5. Verificação (Após fix)

**Checklist de verificação:**
- [ ] Fix testado em ambiente limpo
- [ ] Todos os testes passam
- [ ] Issue original reproduzida e confirmada resolvida
- [ ] Instalação via pip funciona
- [ ] Smoke tests em principais use cases
- [ ] Documentação atualizada
- [ ] CHANGELOG atualizado

**Smoke tests:**
```bash
# Criar venv limpo
python -m venv test_env
source test_env/bin/activate

# Instalar versão com fix
pip install deepbridge==2.0.Y

# Testar imports principais
python -c "from deepbridge import Bridge; print('Core OK')"
python -c "from deepbridge.distillation import KnowledgeDistiller; print('Distillation OK')"
python -c "from deepbridge.synthetic import SyntheticDataGenerator; print('Synthetic OK')"

# Testar caso específico do bug
python reproduce_bug.py
# Deve funcionar sem erro
```

---

### Procedimento para S2 (Alto)

**Processo similar a S1 mas com timelines mais relaxados:**
- Resposta em 8h
- Fix em 3 dias
- Comunicação menos frequente (updates diários)

---

## Rollback de Release

### Quando fazer Rollback?

**Fazer rollback se:**
- Bug S1 sem fix rápido (<4h)
- Múltiplos bugs S2 descobertos
- Instabilidade generalizada
- Perda de dados possível

**NÃO fazer rollback se:**
- Fix rápido (<4h) é viável
- Apenas bugs S3/S4
- Workaround simples existe
- Versão anterior também tem o bug

---

### Processo de Rollback

#### 1. Decisão e Notificação (0-30 min)

```bash
# Notificar equipe
echo "ROLLBACK DECISION: Reverting to v2.0.X due to critical issues in v2.0.Y"

# Criar issue de tracking
gh issue create \
  --title "[ROLLBACK] Reverting v2.0.Y to v2.0.X" \
  --label "rollback,priority:critical"
```

---

#### 2. Yank da Versão Problemática no PyPI (30-60 min)

**⚠️ IMPORTANTE:** "Yank" no PyPI NÃO remove o pacote, apenas o marca como indisponível para novas instalações.

```bash
# Yank versão problemática
# Requer permissões de maintainer no PyPI

# Via web: https://pypi.org/manage/project/deepbridge/releases/
# Ou via API (se disponível)

# Marcar como "yanked" com razão
Reason: "Critical bug - use v2.0.X instead"
```

**Resultado:**
- Usuários com `pip install deepbridge` receberão versão anterior (2.0.X)
- Usuários que já instalaram 2.0.Y NÃO são afetados (precisam downgrade manual)

---

#### 3. Comunicar Downgrade Instructions

**Template:**
```markdown
🚨 URGENT: Please Downgrade DeepBridge

We have identified critical issues in v2.0.Y.

**Action Required:**
```bash
pip install deepbridge==2.0.X
```

**If you experience issues:**
```bash
pip uninstall deepbridge
pip cache purge
pip install deepbridge==2.0.X
```

**Why:** [Brief explanation of bug]
**Status:** We are working on v2.0.Z with fixes. ETA: [date]

**Details:** https://github.com/guhaase/DeepBridge/issues/[n]

We apologize for the inconvenience.
```

---

#### 4. GitHub Release Update

```bash
# Editar release notes da versão problemática
gh release edit v2.0.Y --notes "$(cat <<'EOF'
⚠️ **DO NOT USE THIS VERSION**

This release has been yanked due to critical issues.

**Use v2.0.X instead:**
```bash
pip install deepbridge==2.0.X
```

**Issues:**
- #[n] - [Description]

**Fixed in:** v2.0.Z (coming soon)
EOF
)"
```

---

#### 5. Preparar Fix Proper

```bash
# Trabalhar no fix enquanto usuários usam versão anterior
git checkout -b fix/issues-from-v2.0.Y

# Implementar todos os fixes necessários
# Testar extensivamente
# Preparar v2.0.Z
```

---

## Comunicação de Crise

### Princípios de Comunicação

1. **Transparência:** Admitir o problema claramente
2. **Frequência:** Updates regulares (S1: a cada hora, S2: diariamente)
3. **Ação:** Sempre incluir "o que o usuário deve fazer"
4. **Empatia:** Reconhecer o inconveniente causado
5. **Brevidade:** Ser conciso mas completo

---

### Canais de Comunicação

#### Prioridade 1: GitHub
- Pin da issue
- Update frequente na issue
- Release notes

#### Prioridade 2: Social Media
- Twitter/X
- Reddit (se houver subreddit)
- LinkedIn (posts profissionais)

#### Prioridade 3: Direto
- Email (se mailing list existir)
- Discord/Slack announcements

---

## Templates de Comunicação

### Template: Anúncio de Bug Crítico

```markdown
🚨 CRITICAL BUG - DeepBridge v[X.Y.Z]

**Issue:** [Brief 1-sentence description]

**Impact:**
- Who: [Which users are affected]
- What: [What functionality is broken]
- Severity: S1/S2

**Immediate Action:**
[Workaround or downgrade instructions]

**Status:**
- Discovered: [Timestamp]
- Root cause: [If known, or "Investigating"]
- ETA for fix: [Estimate or "TBD"]

**Tracking:** https://github.com/guhaase/DeepBridge/issues/[n]

We will provide updates every [frequency].
```

---

### Template: Anúncio de Hotfix

```markdown
✅ HOTFIX RELEASED - DeepBridge v[X.Y.Z]

**Fixed Issues:**
- #[n] - [Description]
- #[n] - [Description]

**Action Required:**
```bash
pip install --upgrade deepbridge
```

**Verification:**
```python
import deepbridge
print(deepbridge.__version__)  # Should show [X.Y.Z]
```

**Changes:**
[Brief description of what changed]

**Full Release Notes:** https://github.com/guhaase/DeepBridge/releases/tag/v[X.Y.Z]

Thank you for your patience!
```

---

### Template: Anúncio de Rollback

```markdown
⚠️ ROLLBACK NOTICE - DeepBridge v[X.Y.Z] Yanked

Due to critical issues, we have yanked v[X.Y.Z] from PyPI.

**Action Required - Downgrade:**
```bash
pip install deepbridge==[PREVIOUS_VERSION]
```

**Why:**
[Brief explanation of issues]

**What's Next:**
We are preparing v[NEXT_VERSION] with fixes.
ETA: [Date/Time]

**Apology:**
We sincerely apologize for the disruption. We are improving our testing process to prevent this in the future.

**Track Progress:** https://github.com/guhaase/DeepBridge/issues/[n]
```

---

### Template: Post-Mortem Summary

```markdown
📊 POST-MORTEM: [Incident Name]

**Date:** [YYYY-MM-DD]
**Duration:** [X hours]
**Severity:** S1/S2
**Impact:** [Number of users / % of user base]

## Timeline
- **[HH:MM]** - Issue detected
- **[HH:MM]** - Team notified, investigation started
- **[HH:MM]** - Root cause identified
- **[HH:MM]** - Fix implemented
- **[HH:MM]** - Fix deployed to production
- **[HH:MM]** - Incident resolved

## Root Cause
[Detailed explanation of what went wrong]

## Resolution
[How it was fixed]

## Lessons Learned

**What Went Well:**
- [Thing 1]
- [Thing 2]

**What Went Wrong:**
- [Thing 1]
- [Thing 2]

## Action Items
- [ ] [Action 1] - Assigned: [Name] - Due: [Date]
- [ ] [Action 2] - Assigned: [Name] - Due: [Date]
- [ ] [Action 3] - Assigned: [Name] - Due: [Date]

## Prevention
[Steps being taken to prevent recurrence]

---

Thank you to everyone who helped resolve this incident quickly.
```

---

## Pós-Mortem

### Quando Conduzir Pós-Mortem

**Obrigatório para:**
- Todos os incidentes S1
- Incidentes S2 que afetaram >50% dos usuários
- Qualquer rollback

**Opcional para:**
- Incidentes S2 menores
- Incidentes S3 recorrentes

---

### Processo de Pós-Mortem

#### 1. Reunião de Pós-Mortem (Dentro de 7 dias)

**Participantes:**
- Lead Maintainer (facilitador)
- Technical Lead
- Qualquer pessoa envolvida na resposta

**Agenda:**
1. Timeline do incidente (15 min)
2. Root cause analysis (20 min)
3. O que funcionou / não funcionou (15 min)
4. Action items (10 min)

**Regras:**
- **Blameless:** Foco no processo, não nas pessoas
- **Factual:** Baseado em evidências, não suposições
- **Actionable:** Toda conclusão → action item específico

---

#### 2. Documento de Pós-Mortem

**Estrutura:**
```markdown
# Post-Mortem: [Incident Name]

**Date:** [YYYY-MM-DD]
**Authors:** [Names]
**Status:** Draft / Final

## Executive Summary
[2-3 sentences: what happened, impact, resolution]

## Timeline
[Detailed timeline with timestamps]

## Root Cause Analysis

### What Happened
[Factual description]

### Why It Happened
[Root cause - use "5 Whys" technique]

### Contributing Factors
- [Factor 1]
- [Factor 2]

## Impact Assessment

### Metrics
- **Users Affected:** [Number / Percentage]
- **Duration:** [Hours/Days]
- **Downtime:** [If applicable]
- **Data Loss:** [If any]

### User Impact
[Qualitative description]

## Response Evaluation

### What Went Well
- [Positive 1]
- [Positive 2]

### What Could Be Improved
- [Improvement 1]
- [Improvement 2]

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action 1] | [Name] | [Date] | 🟡 In Progress |
| [Action 2] | [Name] | [Date] | ⬜ Not Started |

## Prevention Measures

### Immediate (Within 1 week)
- [ ] [Measure 1]

### Short-term (Within 1 month)
- [ ] [Measure 2]

### Long-term (Within 3 months)
- [ ] [Measure 3]

## Appendix

### References
- Issue: #[number]
- PR fixes: #[numbers]
- Related incidents: [Links]

### Data
[Logs, metrics, screenshots]
```

---

#### 3. Compartilhar e Arquivar

**Compartilhar:**
- Internamente: Todos os maintainers
- Publicamente: GitHub discussions (opcional, para transparência)
- Blog post (opcional, para incidentes grandes)

**Arquivar:**
- Salvar em `desenvolvimento/postmortems/YYYY-MM-DD-incident-name.md`
- Adicionar ao índice de post-mortems

---

#### 4. Acompanhamento de Action Items

**Tracking:**
```bash
# Criar issues para cada action item
gh issue create \
  --title "[Post-Mortem Action] [Description]" \
  --label "postmortem,improvement" \
  --assignee [owner]

# Adicionar a projeto/milestone
gh issue develop [issue-number] --milestone "Post-Incident Improvements"
```

**Review:**
- Weekly: Check-in em action items
- Monthly: Review de progresso com equipe

---

## Testes de Contingência

### Exercícios de Simulação (Recomendado Quarterly)

**Game Day: Simular Incidente S1**
1. Designar "incident master" que simula bug
2. Equipe responde como em incidente real
3. Medir tempo de detecção → resolução
4. Identificar gaps no plano

**Exemplos de cenários:**
- "PyPI deploy falhou, pacote corrompido"
- "Dependência crítica descontinuada"
- "Vulnerabilidade CVE descoberta no código"

---

## Métricas de Contingência

### KPIs para Rastrear

| Métrica | Target | Medição |
|---------|--------|---------|
| Time to Detect (TTD) | < 1 hour | Tempo até issue criada |
| Time to Respond (TTR) | < 2 hours (S1) | Tempo até primeira ação |
| Time to Resolve (TTRes) | < 24 hours (S1) | Tempo até fix deployed |
| Recurrence Rate | < 5% | % de bugs que retornam |

---

## Contatos de Emergência

### Maintainers

| Nome | Role | Email | Phone | Discord | Timezone |
|------|------|-------|-------|---------|----------|
| [Nome 1] | Lead | [email] | [phone] | [handle] | UTC-X |
| [Nome 2] | Technical | [email] | [phone] | [handle] | UTC-X |
| [Nome 3] | Community | [email] | [phone] | [handle] | UTC-X |

### Serviços Externos

| Serviço | Contato | Uso |
|---------|---------|-----|
| PyPI Support | pypi-admins@python.org | Issues com PyPI |
| GitHub Support | [Link] | Issues com GitHub |
| DNS Provider | [Link] | Se domínio customizado |

---

## Checklist Rápida de Emergência

### ⚡ S1 Quick Response

- [ ] **0-15 min:** Issue criada e pinned, equipe notificada
- [ ] **15-30 min:** Decisão: Hotfix ou Rollback?
- [ ] **30 min:** Comunicado inicial publicado
- [ ] **1h:** Começar implementação de fix
- [ ] **A cada 1h:** Update público
- [ ] **4h:** Se não resolvido, considerar rollback
- [ ] **24h:** Deve estar resolvido ou rollback executado
- [ ] **7 dias:** Post-mortem completo

---

**Este documento deve ser revisado e atualizado trimestralmente.**

**Última revisão:** 2026-02-16
**Próxima revisão:** 2026-05-16
