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
- [ ] README.md atualizado:
  - [ ] Banner de aviso v2.0 adicionado no topo
  - [ ] Links para novos repos (distillation, synthetic)
  - [ ] Link para migration guide
  - [ ] Badges atualizados
- [ ] Exemplos criados em `examples/`:
  - [ ] `examples/robustness_example.py` criado
  - [ ] `examples/fairness_example.py` criado
  - [ ] Exemplo de robustness testado e funcionando
  - [ ] Exemplo de fairness testado e funcionando
- [ ] CHANGELOG.md criado:
  - [ ] Seção [2.0.0-alpha.1] com breaking changes
  - [ ] Lista de removidos (distillation, synthetic)
  - [ ] Lista de mudanças
  - [ ] Link para migration guide
  - [ ] Formato Keep a Changelog seguido
- [ ] Migration guide revisado:
  - [ ] `desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md` existe
  - [ ] Instruções claras de migração
  - [ ] Exemplos de código antes/depois
- [ ] Commit e push realizados

### deepbridge-distillation
- [ ] README.md completo:
  - [ ] Badges adicionados (Tests, codecov, PyPI, Python version)
  - [ ] Descrição clara do pacote
  - [ ] Instruções de instalação
  - [ ] Quick start com exemplo
  - [ ] Lista de features
  - [ ] Link para documentação
  - [ ] Links para repos relacionados (core, synthetic)
  - [ ] Licença mencionada
- [ ] Exemplos criados em `examples/`:
  - [ ] `examples/basic_distillation.py` criado
  - [ ] Exemplo testado e funcionando
  - [ ] Exemplo é autocontido (com geração de dados)
- [ ] CHANGELOG.md criado:
  - [ ] Seção [2.0.0-alpha.1] - Initial release
  - [ ] Migração do DeepBridge v1.x mencionada
  - [ ] Features listadas
- [ ] Commit e push realizados

### deepbridge-synthetic
- [ ] README.md completo:
  - [ ] Badges adicionados
  - [ ] Nota destacando que é standalone
  - [ ] Descrição clara do pacote
  - [ ] Instruções de instalação
  - [ ] Quick start com exemplo
  - [ ] Lista de features
  - [ ] Link para documentação
  - [ ] Links para repos relacionados
  - [ ] Licença mencionada
- [ ] Exemplos criados em `examples/`:
  - [ ] `examples/gaussian_copula_example.py` criado
  - [ ] Exemplo testado e funcionando
  - [ ] Exemplo é autocontido
- [ ] CHANGELOG.md criado:
  - [ ] Seção [2.0.0-alpha.1] - Initial release
  - [ ] Nota sobre standalone
  - [ ] Features listadas
- [ ] Commit e push realizados

### Verificação de Exemplos
- [ ] Exemplo robustness_example.py executado sem erros
- [ ] Exemplo fairness_example.py executado sem erros
- [ ] Exemplo basic_distillation.py executado sem erros
- [ ] Exemplo gaussian_copula_example.py executado sem erros
- [ ] Todos os exemplos geram saída esperada

### Badges (verificar URLs corretas)
- [ ] Badge de Tests aponta para GitHub Actions
- [ ] Badge de codecov aponta para Codecov
- [ ] Badge de PyPI será válido após publicação
- [ ] Badge de Python version correto (3.10+)

### Links (verificar funcionam)
- [ ] Links entre repos funcionam
- [ ] Link para migration guide funciona
- [ ] Links para documentação preparados
- [ ] Links de licença corretos

### Commits
- [ ] Commit do core com mensagem descritiva
- [ ] Commit do distillation com mensagem descritiva
- [ ] Commit do synthetic com mensagem descritiva
- [ ] Todos os commits pushed para GitHub
- [ ] Histórico git limpo e organizado

---

**STATUS DA FASE 4:** ⬜ NÃO INICIADA | 🚧 EM ANDAMENTO | ✅ CONCLUÍDA

**Critério para marcar como CONCLUÍDA:**
- ✅ TODOS os documentos criados e atualizados
- ✅ TODOS os exemplos funcionando
- ✅ TODOS os commits e push realizados
- ✅ Nenhum link quebrado
- ✅ Badges corretos

---

**EXECUTAR AGORA DE FORMA 100% AUTOMÁTICA**
