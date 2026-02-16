# Plano de Contingência - DeepBridge v2.0

Procedimentos de resposta a emergências, bugs críticos e situações inesperadas.

---

## 🎯 Objetivo

Este documento define:
1. Ações para bugs críticos
2. Processo de rollback
3. Templates de comunicação
4. Escalação e responsabilidades
5. SLA e tempos de resposta

---

## 🚨 Classificação de Severidade

### Nível 1: CRÍTICO
**Impacto:** Sistema inutilizável, perda de dados, vulnerabilidade de segurança

**Exemplos:**
- Crash ao importar o pacote
- Perda/corrupção de dados
- Vulnerabilidade de segurança descoberta
- Quebra completa de funcionalidade core

**SLA:** Resposta < 2 horas, Fix < 24 horas

**Ações:**
1. Ativar hotfix workflow imediatamente
2. Notificar todos os usuários
3. Considerar rollback se fix demorar

---

### Nível 2: ALTO
**Impacto:** Funcionalidade importante quebrada, workaround difícil

**Exemplos:**
- Erro em funcionalidade principal
- Performance severely degraded
- Documentação incorreta causando uso errado
- Incompatibilidade com versões comuns de dependências

**SLA:** Resposta < 8 horas, Fix < 72 horas

**Ações:**
1. Priorizar fix
2. Comunicar issue e workaround se disponível
3. Incluir em próximo patch release

---

### Nível 3: MÉDIO
**Impacto:** Funcionalidade secundária afetada, workaround disponível

**Exemplos:**
- Bug em feature opcional
- Performance issue em casos específicos
- Erro de documentação menor
- Edge case não tratado

**SLA:** Resposta < 24 horas, Fix < 1 semana

**Ações:**
1. Adicionar à milestone da próxima release
2. Documentar workaround
3. Responder na issue com plano

---

### Nível 4: BAIXO
**Impacto:** Inconveniência menor, cosmético

**Exemplos:**
- Typo em mensagem de erro
- Warning desnecessário
- Inconsistência de estilo
- Sugestão de melhoria

**SLA:** Resposta < 48 horas, Fix quando conveniente

**Ações:**
1. Adicionar ao backlog
2. Aceitar contribuições da comunidade

---

## 🔥 Procedimentos de Emergência

### Cenário 1: Bug Crítico Descoberto Após Release

**Situação:** v2.0.0 foi lançada, mas usuários reportam crash fatal.

**Procedimento:**

1. **Confirmar severidade (< 30 minutos)**
   ```bash
   # Reproduzir imediatamente
   python -m venv emergency_test
   source emergency_test/bin/activate
   pip install deepbridge==2.0.0
   python reproduce_critical_bug.py
   ```

2. **Avaliar opções (< 1 hora)**
   - **Opção A:** Fix rápido possível → Hotfix
   - **Opção B:** Fix complexo → Rollback + Comunicação

3. **Se HOTFIX:**
   ```bash
   # Branch de hotfix
   git checkout -b hotfix/2.0.1 v2.0.0
   
   # Implementar fix mínimo
   # ... código ...
   
   # Teste rápido mas essencial
   pytest tests/critical/
   
   # Bump versão
   # Atualizar para 2.0.1
   
   # Release imediato
   git tag -a v2.0.1 -m "Critical hotfix"
   python -m build
   twine upload dist/*
   
   # Comunicar
   # (ver templates abaixo)
   ```

4. **Se ROLLBACK:**
   ```bash
   # Yankar release quebrada do PyPI
   # ATENÇÃO: Yank não remove, só marca como não instalável por padrão
   twine upload --repository pypi --skip-existing \
       --config-file ~/.pypirc \
       --comment "Critical bug, use 2.0.1 instead" \
       dist/deepbridge-2.0.0*
   
   # Ou via interface web do PyPI
   # Settings → Manage → Yank
   
   # Comunicar rollback imediatamente
   ```

5. **Comunicação (< 2 horas do descobrimento)**
   - Criar issue no GitHub
   - Postar no Discussions
   - Atualizar README com aviso
   - GitHub Release com nota de urgência

---

### Cenário 2: Dependência Quebrada

**Situação:** Nova versão de PyTorch/Transformers quebra DeepBridge.

**Procedimento:**

1. **Pin versão problemática**
   ```python
   # setup.py ou pyproject.toml
   dependencies = [
       "torch>=1.10.0,<2.1.0",  # Pin max version
       "transformers>=4.20.0,!=4.35.0",  # Exclude broken version
   ]
   ```

2. **Release patch urgente**
   ```bash
   git checkout -b fix/pin-dependency master
   # Atualizar dependencies
   git commit -m "fix: pin dependency to avoid broken version"
   # ... release 2.0.1
   ```

3. **Comunicar workaround**
   ```markdown
   ## Workaround for PyTorch 2.1.0 incompatibility
   
   If you encounter [error], downgrade PyTorch:
   
   ```bash
   pip install torch==2.0.1
   ```
   
   We are working on compatibility with PyTorch 2.1+.
   ```

4. **Trabalhar em compatibility fix**
   - Branch separado
   - Testar extensivamente
   - Release quando pronto

---

### Cenário 3: Segurança Vulnerabilidade

**Situação:** CVE reportado em DeepBridge ou dependência.

**Procedimento:**

1. **Avaliar impacto (URGENTE)**
   - Afeta versões em produção?
   - Exploit público disponível?
   - Severidade (CVSS score)?

2. **Fix silencioso se necessário**
   ```bash
   # NÃO criar issue pública inicialmente se exploit grave
   # Fix em branch privado
   git checkout -b security/CVE-2025-XXXX master
   
   # Implementar fix
   # ...
   
   # Release hotfix
   git tag -a v2.0.1 -m "Security fix"
   # ... publish
   ```

3. **Disclosure responsável**
   - Aguardar 24-48h após fix publicado
   - Então publicar advisory no GitHub
   - Creditar reporter (se autorizado)

4. **Comunicação**
   ```markdown
   # Security Advisory: [Título]
   
   **Severity:** High
   **Affected versions:** 2.0.0
   **Fixed in:** 2.0.1
   
   ## Description
   [Descrição técnica]
   
   ## Impact
   [O que atacante pode fazer]
   
   ## Mitigation
   Upgrade immediately:
   ```bash
   pip install --upgrade deepbridge
   ```
   
   ## Credit
   Thanks to [researcher] for responsible disclosure.
   ```

---

## 🔄 Processo de Rollback

### Quando Fazer Rollback

**Critérios:**
- ✅ Bug crítico afeta >50% dos usuários
- ✅ Sem fix rápido disponível (>24h estimado)
- ✅ Versão anterior estável disponível
- ❌ Não fazer rollback se breaking changes já adotados

### Como Fazer Rollback (PyPI)

**IMPORTANTE:** PyPI não permite deletar releases. Apenas "yank" (ocultar).

```bash
# 1. Yank release problemática via interface web
# https://pypi.org/manage/project/deepbridge/release/2.0.0/

# 2. Ou via twine (se suportado):
twine upload --skip-existing \
    --comment "Critical bug, use 1.9.9 instead" \
    dist/deepbridge-2.0.0*

# 3. Comunicar claramente
```

### Como Fazer Rollback (Git)

```bash
# Opção 1: Revert commits (preferido)
git revert HEAD~3..HEAD  # Reverte últimos 3 commits
git push origin master

# Opção 2: Reset (apenas se não publicado)
git reset --hard HEAD~3
git push --force origin master  # ⚠️ Cuidado!

# Opção 3: Criar branch de fix baseado em versão antiga
git checkout -b fix-from-stable v1.9.9
# ... trabalhar no fix
```

---

## 📢 Templates de Comunicação

### Template 1: Anúncio de Bug Crítico

```markdown
# ⚠️ Critical Issue in v2.0.0

We have identified a critical issue in DeepBridge v2.0.0 that causes [descrição breve].

## Impact
- [Quem é afetado]
- [O que não funciona]

## Status
We are working on a hotfix and expect to release v2.0.1 within [timeframe].

## Workaround
Until the fix is released, please:
```bash
[workaround se disponível]
```

## Updates
We will update this issue with progress. 

**ETA for fix:** [data/hora]

We apologize for the inconvenience and appreciate your patience.

---
**Reported:** [timestamp]
**Severity:** Critical
**Tracking:** #[issue number]
```

### Template 2: Anúncio de Hotfix

```markdown
# 🚀 Hotfix Release: v2.0.1

We have released v2.0.1 to address the critical issue reported in #[issue].

## What Changed
- Fixed: [descrição do bug]
- Impact: [quem estava afetado]

## Upgrade Instructions
```bash
pip install --upgrade deepbridge
# Verify
python -c "import deepbridge; print(deepbridge.__version__)"
# Should print: 2.0.1
```

## Details
[Descrição técnica do problema e solução]

## Testing
This release has been tested with:
- [cenário 1]
- [cenário 2]

Thank you for your patience and for reporting this issue.

---
**Released:** [timestamp]
**Fixes:** #[issue]
```

### Template 3: Anúncio de Rollback

```markdown
# ⚠️ Rollback Notice: v2.0.0 Yanked

Due to critical issues, we have yanked v2.0.0 from PyPI.

## What Happened
[Explicação clara do problema]

## Action Required
If you installed v2.0.0, please downgrade:
```bash
pip install deepbridge==1.9.9
```

## Next Steps
We are working on a fixed v2.0.1 release. We will announce when it's ready.

## Apology
We sincerely apologize for this disruption. We are reviewing our release process to prevent this in the future.

---
**Yanked:** [timestamp]
**Recommended version:** 1.9.9
**Tracking:** #[issue]
```

### Template 4: Security Advisory

```markdown
# 🔒 Security Advisory: [CVE-ID]

**Severity:** [Low/Medium/High/Critical]
**Affected versions:** [range]
**Fixed in:** [version]

## Summary
[Descrição não-técnica do problema]

## Technical Details
[Descrição técnica]

## Exploitation
[Como pode ser explorado - se apropriado]

## Impact
[O que um atacante pode fazer]

## Remediation
Upgrade immediately to v[fixed version]:
```bash
pip install --upgrade deepbridge
```

## Workaround
If you cannot upgrade immediately:
[workaround se disponível]

## Timeline
- **Discovered:** [date]
- **Fixed:** [date]
- **Released:** [date]
- **Disclosed:** [date]

## Credit
[Se aplicável] Thanks to [researcher/organization] for responsible disclosure.

---
**CVE:** [CVE-ID]
**CVSS Score:** [score]
**References:** [links]
```

---

## 👥 Responsabilidades e Escalação

### Responsáveis

**Maintainer Principal:**
- Decisões finais sobre hotfixes
- Aprovação de rollbacks
- Comunicação oficial

**Contributors:**
- Triagem inicial de bugs
- Implementação de fixes
- Code review

**Community:**
- Report de bugs
- Teste de fixes
- Sugestões

### Escalação

**Nível 4 (Baixo):**
→ Qualquer contributor pode resolver

**Nível 3 (Médio):**
→ Contributor experiente + review

**Nível 2 (Alto):**
→ Maintainer + review obrigatório

**Nível 1 (Crítico):**
→ Maintainer principal + decisão imediata

---

## ⏱️ SLA (Service Level Agreement)

### Tempos de Resposta

| Severidade | Primeira Resposta | Fix Estimado | Patch Release |
|------------|------------------|--------------|---------------|
| Crítico    | < 2 horas        | < 24 horas   | Imediato      |
| Alto       | < 8 horas        | < 72 horas   | < 1 semana    |
| Médio      | < 24 horas       | < 1 semana   | Próximo patch |
| Baixo      | < 48 horas       | Backlog      | Quando possível |

**Nota:** SLAs são metas, não garantias. Projetos open source dependem de disponibilidade de voluntários.

---

## 📊 Métricas de Incidentes

Acompanhar para melhorar processos:

- **MTTD** (Mean Time To Detect): Tempo até descobrir bug
- **MTTR** (Mean Time To Respond): Tempo até primeira resposta
- **MTTF** (Mean Time To Fix): Tempo até fix disponível
- **Número de hotfixes** por release
- **Número de rollbacks** por ano

---

## 🧪 Testing de Emergência

### Smoke Tests Mínimos

```bash
# Quick sanity check antes de hotfix release
python -c "import deepbridge; print(deepbridge.__version__)"
python -c "from deepbridge.core import BridgeConfig"
python -c "from deepbridge_distillation import DistillationTrainer"
python -c "from deepbridge_synthetic import SyntheticDataGenerator"
```

### Testes Críticos

```bash
# Suite essencial (deve rodar em <5 min)
pytest tests/critical/ -v --tb=short
```

---

## 📝 Post-Mortem

Após resolver incidente crítico, documentar:

**Template de Post-Mortem:**
```markdown
# Post-Mortem: [Título do Incidente]

**Data:** [data]
**Severidade:** [nível]
**Duração:** [tempo até resolução]

## O Que Aconteceu
[Descrição cronológica]

## Causa Raiz
[Análise técnica]

## Impacto
- [Número de usuários afetados]
- [Funcionalidades afetadas]
- [Downtime se aplicável]

## Linha do Tempo
- [HH:MM] Incidente reportado
- [HH:MM] Confirmado e classificado
- [HH:MM] Hotfix iniciado
- [HH:MM] Fix testado
- [HH:MM] Hotfix released
- [HH:MM] Verificado resolvido

## O Que Foi Bem
[Aspectos positivos da resposta]

## O Que Pode Melhorar
[Oportunidades de melhoria]

## Action Items
- [ ] [Ação 1] - @responsável
- [ ] [Ação 2] - @responsável

## Lições Aprendidas
[Conclusões]
```

---

## 🔗 Contatos e Recursos

### Documentação de Emergência
- Este documento: `PLANO_CONTINGENCIA.md`
- Workflow de bugfix: `WORKFLOW_BUGFIX.md`
- Workflow de release: `WORKFLOW_RELEASE.md`

### Links Úteis
- GitHub Issues: [URL]
- PyPI Project: https://pypi.org/project/deepbridge/
- CI/CD: [URL]
- Monitoring: [URL se aplicável]

---

**Última atualização:** 2025-02-16

**Revisão:** Este plano deve ser revisado após cada incidente crítico e no mínimo trimestralmente.
