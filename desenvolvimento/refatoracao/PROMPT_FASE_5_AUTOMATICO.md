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
