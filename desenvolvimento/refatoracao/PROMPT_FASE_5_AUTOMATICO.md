# PROMPT PARA EXECUÇÃO AUTOMÁTICA - FASE 5: Release v2.0

**⚠️ ATENÇÃO:** Esta fase requer credenciais do PyPI e não pode ser 100% automática.

---

## 🎯 OBJETIVO

Publicar v2.0.0 no PyPI e anunciar o release:
- Criar tags de versão
- Publicar no Test PyPI (requer token)
- Testar instalação
- Publicar no PyPI oficial (requer token)
- Criar GitHub Releases
- Anunciar mudanças
- Deprecar v1.x

---

## 📋 TAREFAS A EXECUTAR

### Tarefa 1: Preparar Release Candidate (AUTOMÁTICO)
- Atualizar versões para `2.0.0-rc.1` em todos os repos
- Criar commits
- Criar tags `v2.0.0-rc.1`
- Push tags

### Tarefa 2: Build dos pacotes (AUTOMÁTICO)
- Executar `poetry build` em cada repo
- Verificar que dist/ foi criado

### Tarefa 3: Publicar no Test PyPI (⚠️ REQUER TOKEN)
**AÇÃO MANUAL NECESSÁRIA:**
```bash
# Usuário deve configurar token primeiro:
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi pypi-YOUR_TEST_TOKEN_HERE

# Depois, pode executar:
poetry publish -r testpypi
```

### Tarefa 4: Testar instalação do Test PyPI (AUTOMÁTICO)
- Criar venv temporário
- Instalar de Test PyPI
- Testar imports
- Limpar venv

### Tarefa 5: Release final no PyPI (⚠️ REQUER TOKEN)
**AÇÃO MANUAL NECESSÁRIA:**
- Atualizar versões para `2.0.0` (sem -rc)
- Criar tags `v2.0.0`
- Configurar token PyPI
- Publicar

### Tarefa 6: Criar GitHub Releases (SEMI-AUTOMÁTICO)
- Criar release notes
- Publicar via `gh release create`

### Tarefa 7: Deprecar v1.x (AUTOMÁTICO)
- Adicionar deprecation warning no v1.x
- Bump para v1.63.0
- Publicar última versão v1.x

---

## ⚙️ EXECUÇÃO

**PARTE AUTOMÁTICA:**
Posso executar automaticamente:
1. ✅ Atualizar versões
2. ✅ Criar tags
3. ✅ Build com poetry
4. ✅ Testar instalações
5. ✅ Criar release notes
6. ✅ Deprecar v1.x

**PARTE MANUAL (requer usuário):**
1. ⚠️ Configurar tokens PyPI
2. ⚠️ Executar `poetry publish`
3. ⚠️ Verificar no PyPI web

---

## 🔧 MODO DE EXECUÇÃO RECOMENDADO

### Opção 1: Execução híbrida (RECOMENDADO)
1. Execute automaticamente até a Tarefa 2 (build)
2. **PAUSE** - Usuário configura tokens PyPI
3. Execute Tarefa 3 (publish Test PyPI)
4. Execute automaticamente Tarefa 4 (testar)
5. **PAUSE** - Usuário valida Test PyPI
6. Execute Tarefa 5 (release final)
7. Execute automaticamente Tarefas 6-7

### Opção 2: Somente preparação (100% AUTOMÁTICO)
Execute apenas Tarefas 1-2 e 6-7:
- Preparar versões
- Build pacotes
- Criar release notes
- Deprecar v1.x
- **Deixar publicação PyPI para usuário fazer manualmente**

---

## 🔍 VERIFICAÇÃO FINAL

Ao finalizar, confirme que:
1. ✅ Versões atualizadas (2.0.0-rc.1 → 2.0.0)
2. ✅ Tags criadas em todos os repos
3. ⚠️ Publicado no Test PyPI (manual)
4. ✅ Testado instalação do Test PyPI
5. ⚠️ Publicado no PyPI oficial (manual)
6. ✅ GitHub Releases criados
7. ✅ v1.x com deprecation warning
8. ✅ v1.63.0 publicado

---

## 📝 REFERÊNCIA

Para detalhes completos, consulte:
`/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/FASE_5_RELEASE.md`

---

## ❓ PERGUNTA PARA O USUÁRIO

**Qual modo de execução você prefere?**

A) **Híbrido** - Execute automaticamente o máximo possível, pausando quando precisar de credenciais PyPI

B) **Somente preparação** - Execute apenas preparação automática, deixe publicação PyPI para manual

C) **Totalmente manual** - Apenas forneça checklist, você executa tudo manualmente

Por favor, responda A, B ou C para prosseguir.

---

## ✅ CHECKLIST FINAL - VERIFICAR APÓS EXECUÇÃO

### Preparação Release Candidate
- [ ] Versões atualizadas para `2.0.0-rc.1`:
  - [ ] deepbridge/__init__.py: `__version__ = '2.0.0-rc.1'`
  - [ ] deepbridge/pyproject.toml: `version = "2.0.0-rc.1"`
  - [ ] deepbridge-distillation/__init__.py: `__version__ = '2.0.0-rc.1'`
  - [ ] deepbridge-distillation/pyproject.toml: `version = "2.0.0-rc.1"`
  - [ ] deepbridge-synthetic/__init__.py: `__version__ = '2.0.0-rc.1'`
  - [ ] deepbridge-synthetic/pyproject.toml: `version = "2.0.0-rc.1"`
- [ ] Commits criados para cada repo
- [ ] Tags criadas:
  - [ ] deepbridge: `v2.0.0-rc.1`
  - [ ] deepbridge-distillation: `v2.0.0-rc.1`
  - [ ] deepbridge-synthetic: `v2.0.0-rc.1`
- [ ] Tags pushed para GitHub

### Build dos Pacotes
- [ ] deepbridge: `poetry build` executado
- [ ] deepbridge: `dist/` criado com .whl e .tar.gz
- [ ] deepbridge-distillation: `poetry build` executado
- [ ] deepbridge-distillation: `dist/` criado
- [ ] deepbridge-synthetic: `poetry build` executado
- [ ] deepbridge-synthetic: `dist/` criado

### Test PyPI (⚠️ Requer configuração manual)
- [ ] Repositório Test PyPI configurado: `poetry config repositories.testpypi ...`
- [ ] Token Test PyPI configurado: `poetry config pypi-token.testpypi ...`
- [ ] deepbridge publicado no Test PyPI
- [ ] deepbridge-distillation publicado no Test PyPI
- [ ] deepbridge-synthetic publicado no Test PyPI
- [ ] URLs verificados:
  - [ ] https://test.pypi.org/project/deepbridge/
  - [ ] https://test.pypi.org/project/deepbridge-distillation/
  - [ ] https://test.pypi.org/project/deepbridge-synthetic/

### Testes de Instalação (Test PyPI)
- [ ] Ambiente virtual criado para teste
- [ ] deepbridge instalado do Test PyPI
- [ ] deepbridge imports testados:
  - [ ] `import deepbridge`
  - [ ] `from deepbridge import DBDataset, Experiment`
- [ ] deepbridge-distillation instalado do Test PyPI
- [ ] deepbridge-distillation imports testados:
  - [ ] `import deepbridge_distillation`
  - [ ] `from deepbridge_distillation import AutoDistiller`
  - [ ] `import deepbridge` funciona (dependência)
- [ ] deepbridge-synthetic instalado do Test PyPI
- [ ] deepbridge-synthetic imports testados:
  - [ ] `import deepbridge_synthetic`
  - [ ] `from deepbridge_synthetic import Synthesize`
- [ ] Ambiente de teste limpo

### Release Final (⚠️ Requer configuração manual)
- [ ] Versões atualizadas para `2.0.0` (sem -rc)
- [ ] Tags finais criadas:
  - [ ] deepbridge: `v2.0.0`
  - [ ] deepbridge-distillation: `v2.0.0`
  - [ ] deepbridge-synthetic: `v2.0.0`
- [ ] Tags pushed para GitHub
- [ ] Token PyPI oficial configurado
- [ ] deepbridge publicado no PyPI oficial
- [ ] deepbridge-distillation publicado no PyPI oficial
- [ ] deepbridge-synthetic publicado no PyPI oficial
- [ ] URLs verificados:
  - [ ] https://pypi.org/project/deepbridge/
  - [ ] https://pypi.org/project/deepbridge-distillation/
  - [ ] https://pypi.org/project/deepbridge-synthetic/

### GitHub Releases
- [ ] GitHub Release criado para deepbridge v2.0.0:
  - [ ] Título descritivo
  - [ ] Release notes completo
  - [ ] Menção de breaking changes
  - [ ] Links para migration guide
  - [ ] Links para novos repos
- [ ] GitHub Release criado para deepbridge-distillation v2.0.0
- [ ] GitHub Release criado para deepbridge-synthetic v2.0.0
- [ ] Todos os releases visíveis no GitHub

### Deprecação v1.x
- [ ] Branch master (v1.x) checked out
- [ ] Deprecation warning adicionado em deepbridge/__init__.py
- [ ] Versão atualizada para `1.63.0`
- [ ] Commit e tag `v1.63.0` criados
- [ ] v1.63.0 publicado no PyPI
- [ ] Usuários verão warning ao importar v1.x

### Anúncios
- [ ] Post criado no GitHub Discussions
- [ ] README.md atualizado com link para release
- [ ] Documentação atualizada (se aplicável)
- [ ] Twitter/LinkedIn/Blog atualizado (se aplicável)

### Verificações Finais
- [ ] Instalação funciona: `pip install deepbridge`
- [ ] Instalação funciona: `pip install deepbridge-distillation`
- [ ] Instalação funciona: `pip install deepbridge-synthetic`
- [ ] PyPI mostra versão 2.0.0 para todos
- [ ] Download counts iniciando
- [ ] Nenhum erro crítico reportado

---

**STATUS DA FASE 5:** ⬜ NÃO INICIADA | 🚧 EM ANDAMENTO | ✅ CONCLUÍDA

**Critério para marcar como CONCLUÍDA:**
- ✅ Todos os 3 pacotes publicados no PyPI oficial
- ✅ Versão 2.0.0 disponível para download
- ✅ GitHub Releases criados
- ✅ v1.x deprecado
- ✅ Testes de instalação passando

⚠️ **Esta fase requer tokens PyPI - não pode ser 100% automática**

---

**MODO RECOMENDADO:** Híbrido (Opção A)
