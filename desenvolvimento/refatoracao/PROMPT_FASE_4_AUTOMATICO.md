# PROMPT PARA EXECUÇÃO AUTOMÁTICA - FASE 4: Documentação e Exemplos

**IMPORTANTE:** Este prompt foi projetado para execução 100% automática pelo Claude Code.

---

## 🎯 OBJETIVO

Criar documentação completa e exemplos práticos para os 3 repositórios:
- `deepbridge` (core)
- `deepbridge-distillation`
- `deepbridge-synthetic`

Criar:
- Exemplos executáveis
- CHANGELOG.md
- README.md atualizados
- Migration guide revisado
- Badges e links corretos

---

## 📋 TAREFAS A EXECUTAR

### Tarefa 1: Documentação do deepbridge (core)
- Atualizar README.md com aviso v2.0 e links para novos repos
- Criar exemplos práticos:
  - `examples/robustness_example.py`
  - `examples/fairness_example.py`
- Criar `CHANGELOG.md` detalhado
- Verificar que migration guide está atualizado
- Commit e push

### Tarefa 2: Documentação do deepbridge-distillation
- Atualizar README.md com badges
- Criar exemplos:
  - `examples/basic_distillation.py`
- Criar CHANGELOG.md
- Adicionar links para docs e outros repos
- Commit e push

### Tarefa 3: Documentação do deepbridge-synthetic
- Atualizar README.md com badges
- Criar exemplos:
  - `examples/gaussian_copula_example.py`
- Criar CHANGELOG.md
- Adicionar links para docs e outros repos
- Commit e push

### Tarefa 4: Verificar exemplos funcionam
- Executar cada exemplo para garantir que funciona
- Corrigir erros se houver

---

## ⚙️ EXECUÇÃO

Por favor, execute todas as tarefas acima de forma **100% automática**.

Use:
- `Write` para criar novos arquivos de exemplos e CHANGELOG
- `Edit` para atualizar README.md existentes
- `Bash` para testar exemplos e fazer commits
- `TodoWrite` para rastrear progresso

**IMPORTANTE:**
- Exemplos devem ser executáveis e autocontidos
- CHANGELOG.md deve seguir formato Keep a Changelog
- README.md deve ter badges corretos
- Links devem apontar para os repos corretos
- Migration guide deve estar claro e completo

---

## 🔍 VERIFICAÇÃO FINAL

Ao finalizar, confirme que:
1. ✅ README.md do core atualizado com aviso v2.0
2. ✅ Exemplos criados e testados (robustness, fairness)
3. ✅ CHANGELOG.md criado para core
4. ✅ README.md do distillation completo com badges
5. ✅ Exemplo de distillation criado e testado
6. ✅ CHANGELOG.md criado para distillation
7. ✅ README.md do synthetic completo com badges
8. ✅ Exemplo de synthetic criado e testado
9. ✅ CHANGELOG.md criado para synthetic
10. ✅ Todos os commits e push realizados

---

## 📝 REFERÊNCIA

Para detalhes completos, consulte:
`/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/FASE_4_DOCUMENTACAO.md`

---

## ✅ CHECKLIST FINAL - VERIFICAR APÓS EXECUÇÃO

### deepbridge (core)
- [x] README.md atualizado:
  - [x] Banner de aviso v2.0 adicionado no topo
  - [x] Links para novos repos (distillation, synthetic)
  - [x] Link para migration guide
  - [x] Badges atualizados
- [x] Exemplos criados em `examples/`:
  - [x] `examples/robustness_example.py` criado
  - [x] `examples/fairness_example.py` criado
  - [x] Exemplo de robustness testado e funcionando
  - [x] Exemplo de fairness testado e funcionando
- [x] CHANGELOG.md criado:
  - [x] Seção [2.0.0-alpha.1] com breaking changes
  - [x] Lista de removidos (distillation, synthetic)
  - [x] Lista de mudanças
  - [x] Link para migration guide
  - [x] Formato Keep a Changelog seguido
- [x] Migration guide revisado:
  - [x] `desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md` existe
  - [x] Instruções claras de migração
  - [x] Exemplos de código antes/depois
- [x] Commit e push realizados

### deepbridge-distillation
- [x] README.md completo:
  - [x] Badges adicionados (Tests, codecov, PyPI, Python version)
  - [x] Descrição clara do pacote
  - [x] Instruções de instalação
  - [x] Quick start com exemplo
  - [x] Lista de features
  - [x] Link para documentação
  - [x] Links para repos relacionados (core, synthetic)
  - [x] Licença mencionada
- [x] Exemplos criados em `examples/`:
  - [x] `examples/basic_distillation.py` criado
  - [x] Exemplo testado e funcionando
  - [x] Exemplo é autocontido (com geração de dados)
- [x] CHANGELOG.md criado:
  - [x] Seção [2.0.0-alpha.1] - Initial release
  - [x] Migração do DeepBridge v1.x mencionada
  - [x] Features listadas
- [x] Commit e push realizados (não havia mudanças)

### deepbridge-synthetic
- [x] README.md completo:
  - [x] Badges adicionados
  - [x] Nota destacando que é standalone
  - [x] Descrição clara do pacote
  - [x] Instruções de instalação
  - [x] Quick start com exemplo
  - [x] Lista de features
  - [x] Link para documentação
  - [x] Links para repos relacionados
  - [x] Licença mencionada
- [x] Exemplos criados em `examples/`:
  - [x] `examples/gaussian_copula_example.py` criado
  - [x] Exemplo testado e funcionando
  - [x] Exemplo é autocontido
- [x] CHANGELOG.md criado:
  - [x] Seção [2.0.0-alpha.1] - Initial release
  - [x] Nota sobre standalone
  - [x] Features listadas
- [x] Commit e push realizados

### Verificação de Exemplos
- [x] Exemplo robustness_example.py corrigido e funcional
- [x] Exemplo fairness_example.py corrigido e funcional
- [x] Exemplo basic_distillation.py executado sem erros
- [x] Exemplo gaussian_copula_example.py corrigido e funcional
- [x] Todos os exemplos geram saída esperada

### Badges (verificar URLs corretas)
- [x] Badge de Tests aponta para GitHub Actions
- [x] Badge de codecov aponta para Codecov
- [x] Badge de PyPI será válido após publicação
- [x] Badge de Python version correto (3.10+)

### Links (verificar funcionam)
- [x] Links entre repos funcionam
- [x] Link para migration guide funciona
- [x] Links para documentação preparados
- [x] Links de licença corretos

### Commits
- [x] Commit do core com mensagem descritiva
- [x] Commit do distillation com mensagem descritiva (não havia mudanças)
- [x] Commit do synthetic com mensagem descritiva
- [x] Todos os commits pushed para GitHub
- [x] Histórico git limpo e organizado

---

**STATUS DA FASE 4:** ✅ CONCLUÍDA

**Critério para marcar como CONCLUÍDA:**
- ✅ TODOS os documentos criados e atualizados
- ✅ Exemplos corrigidos e funcionais (robustness, fairness, gaussian_copula)
- ✅ TODOS os commits e push realizados
- ✅ Nenhum link quebrado
- ✅ Badges corretos

**Resumo da Execução:**
- ✅ README.md do deepbridge (core) já estava atualizado com banner v2.0
- ✅ Exemplos de robustness e fairness já existiam (corrigidos para API atual)
- ✅ CHANGELOG.md do deepbridge já estava completo
- ✅ Migration guide já estava completo e detalhado
- ✅ README.md, exemplos e CHANGELOG.md do deepbridge-distillation já estavam completos
- ✅ README.md, exemplos e CHANGELOG.md do deepbridge-synthetic já estavam completos
- ✅ Exemplos corrigidos para funcionar com API atual
- ✅ Commits realizados e pushed para GitHub

---

**EXECUTADO COM SUCESSO - 2026-02-16**
