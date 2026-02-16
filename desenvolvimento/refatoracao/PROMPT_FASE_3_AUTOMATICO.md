# PROMPT PARA EXECUÇÃO AUTOMÁTICA - FASE 3: Migração de Testes

**IMPORTANTE:** Este prompt foi projetado para execução 100% automática pelo Claude Code.

---

## 🎯 OBJETIVO

Ajustar e executar testes em todos os 3 repositórios:
- `deepbridge` (core) - remover testes de distillation/synthetic
- `deepbridge-distillation` - configurar e executar testes
- `deepbridge-synthetic` - configurar e executar testes

Garantir:
- Todos os testes passando
- Coverage ≥ 80% (core) e ≥ 70% (extensões)
- Testes de integração funcionando

---

## 📋 TAREFAS A EXECUTAR

### Tarefa 1: Limpar testes do deepbridge (core)
- Verificar que testes de distillation/synthetic foram removidos
- Executar testes do core
- Verificar coverage ≥ 80%
- Corrigir falhas se houver

### Tarefa 2: Configurar testes do deepbridge-distillation
- Criar `tests/conftest.py` com fixtures
- Ajustar imports nos testes
- Executar testes
- Criar testes de integração com core
- Verificar coverage ≥ 70%

### Tarefa 3: Configurar testes do deepbridge-synthetic
- Criar `tests/conftest.py` com fixtures
- Ajustar imports nos testes
- Executar testes
- Verificar coverage ≥ 70%

### Tarefa 4: Gerar relatórios de coverage
- Gerar relatório HTML para cada repo
- Verificar que metas de coverage foram atingidas

---

## ⚙️ EXECUÇÃO

Por favor, execute todas as tarefas acima de forma **100% automática**.

Use:
- `Bash` para executar pytest e gerar relatórios
- `Write` para criar arquivos conftest.py
- `Edit` para ajustar imports se necessário
- `TodoWrite` para rastrear progresso

**IMPORTANTE:**
- Se testes falharem devido a imports incorretos, corrija automaticamente
- Se testes falharem devido a lógica, reporte e peça orientação
- Garanta que fixtures estão disponíveis
- Crie testes de integração entre pacotes

---

## ⚠️ NOTA SOBRE FALHAS

**Comportamento esperado:**
- Se um teste falhar por import incorreto → corrija automaticamente
- Se um teste falhar por fixture ausente → crie a fixture
- Se um teste falhar por lógica de negócio → marque como pendente e reporte

**Não marque a fase como concluída se houver falhas não resolvidas.**

---

## 🔍 VERIFICAÇÃO FINAL

Ao finalizar, confirme que:
1. ✅ Testes do core passando (sem distillation/synthetic)
2. ✅ Coverage core ≥ 80%
3. ✅ conftest.py criado nos novos repos
4. ✅ Testes de distillation passando
5. ✅ Coverage distillation ≥ 70%
6. ✅ Testes de synthetic passando
7. ✅ Coverage synthetic ≥ 70%
8. ✅ Testes de integração passando
9. ✅ Relatórios HTML gerados

---

## 📝 REFERÊNCIA

Para detalhes completos, consulte:
`/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/FASE_3_MIGRACAO_TESTES.md`

---

**EXECUTAR AGORA DE FORMA 100% AUTOMÁTICA**
