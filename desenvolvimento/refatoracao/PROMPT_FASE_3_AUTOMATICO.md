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

## ✅ CHECKLIST FINAL - VERIFICAR APÓS EXECUÇÃO

### deepbridge (core)
- [ ] Testes de distillation removidos (tests/test_distillation/)
- [ ] Testes de synthetic removidos (tests/test_synthetic/)
- [ ] Arquivo `tests/test_utils/test_synthetic_data.py` removido
- [ ] Arquivo `tests/conftest.py` atualizado (sem fixtures de distillation/synthetic)
- [ ] Imports incorretos removidos de conftest.py
- [ ] Testes executados com sucesso: `pytest tests/ -v`
- [ ] Nenhum teste falhando
- [ ] Coverage verificado: `pytest tests/ --cov=deepbridge`
- [ ] Coverage ≥ 80% alcançado
- [ ] Relatório HTML gerado em `htmlcov/index.html`

### deepbridge-distillation
- [ ] Arquivo `tests/conftest.py` criado
- [ ] Fixtures disponíveis:
  - [ ] `sample_data`
  - [ ] `sample_dataset`
  - [ ] `simple_model`
- [ ] Imports ajustados em todos os testes
- [ ] Imports corretos do core: `from deepbridge import DBDataset`
- [ ] Dependências instaladas: `pip install -e .[dev]`
- [ ] Testes executados: `pytest tests/ -v`
- [ ] Todos os testes passando
- [ ] Coverage verificado: `pytest tests/ --cov=deepbridge_distillation`
- [ ] Coverage ≥ 70% alcançado
- [ ] Relatório HTML gerado em `htmlcov/index.html`
- [ ] Teste de integração criado: `tests/integration/test_core_integration.py`
- [ ] Teste de integração passando

### deepbridge-synthetic
- [ ] Arquivo `tests/conftest.py` criado
- [ ] Fixtures disponíveis:
  - [ ] `sample_data`
  - [ ] `numeric_data`
  - [ ] `mixed_data`
- [ ] Imports ajustados em todos os testes
- [ ] Dependências instaladas: `pip install -e .[dev]`
- [ ] Testes executados: `pytest tests/ -v`
- [ ] Todos os testes passando
- [ ] Coverage verificado: `pytest tests/ --cov=deepbridge_synthetic`
- [ ] Coverage ≥ 70% alcançado
- [ ] Relatório HTML gerado em `htmlcov/index.html`

### Relatórios de Coverage
- [ ] Relatório core gerado e visualizado
- [ ] Relatório distillation gerado e visualizado
- [ ] Relatório synthetic gerado e visualizado
- [ ] Todas as metas de coverage atingidas

### Correções Realizadas (se necessário)
- [ ] Falhas de import corrigidas automaticamente
- [ ] Fixtures ausentes criadas
- [ ] Problemas de lógica identificados e reportados (se houver)

### Verificações Finais
- [ ] Total de testes no core: _____ (todos passando)
- [ ] Total de testes em distillation: _____ (todos passando)
- [ ] Total de testes em synthetic: _____ (todos passando)
- [ ] Nenhum warning crítico nos testes
- [ ] Coverage reports acessíveis via browser

---

**STATUS DA FASE 3:** ⬜ NÃO INICIADA | 🚧 EM ANDAMENTO | ✅ CONCLUÍDA

**Critério para marcar como CONCLUÍDA:**
- ✅ TODOS os testes passando em TODOS os repos
- ✅ Coverage mínima atingida (80% core, 70% extensões)
- ✅ Nenhuma falha não resolvida

⚠️ **NÃO marque como concluída se houver testes falhando por problemas de lógica!**

---

**EXECUTAR AGORA DE FORMA 100% AUTOMÁTICA**
