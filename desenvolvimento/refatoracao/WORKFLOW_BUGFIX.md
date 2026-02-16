# Workflow de Bugfix - DeepBridge v2.0

Processo estruturado para correção de bugs, hotfixes e patch releases.

---

## 🎯 Visão Geral

Este documento define o workflow para:
1. **Bugfixes regulares:** correções que vão na próxima release
2. **Hotfixes:** correções urgentes que exigem release imediato
3. **Patch releases:** lançamento de versões de correção (2.0.1, 2.0.2, etc.)

---

## 🐛 Workflow de Bugfix Regular

### 1. Receber e Triar Bug Report

**Ao receber uma issue de bug:**

1. Adicionar label `bug`
2. Verificar se é duplicata
3. Tentar reproduzir o bug
4. Avaliar prioridade:
   - `priority: critical` - quebra funcionalidade essencial, segurança
   - `priority: high` - impacta muitos usuários
   - `priority: medium` - impacta alguns usuários
   - `priority: low` - edge case, workaround disponível

**Template de resposta inicial:**
```markdown
Obrigado por reportar! Vou investigar e retornar em breve.

**Status:** Em análise
**Prioridade:** [a definir]
```

### 2. Reproduzir Bug Localmente

```bash
# Criar ambiente limpo
python -m venv test_env
source test_env/bin/activate  # ou test_env\Scripts\activate no Windows

# Instalar versão reportada
pip install deepbridge==2.0.0  # versão específica do report

# Executar código do report
python reproduce_bug.py
```

**Documentar:**
- ✅ Bug confirmado?
- 📝 Passos para reproduzir
- 🔍 Causa raiz identificada
- 💡 Possível solução

### 3. Criar Branch de Fix

```bash
# Atualizar main
git checkout master
git pull origin master

# Criar branch fix/
git checkout -b fix/issue-123-description

# Exemplo:
git checkout -b fix/issue-123-import-error
```

**Convenção de nomes:**
- `fix/issue-{número}-{descrição-curta}`
- `fix/memory-leak-dataloader`
- `fix/cuda-out-of-memory`

### 4. Implementar Fix

**Boas práticas:**

1. **Fix mínimo:** altere apenas o necessário
2. **Comentários:** explique por que o fix funciona
3. **Compatibilidade:** não quebre APIs existentes
4. **Performance:** não degrade performance

**Exemplo:**
```python
# Antes (buggy)
def process_data(data):
    return data.split(",")  # Bug: falha se data é None

# Depois (fixed)
def process_data(data):
    # Fix: handle None input gracefully (issue #123)
    if data is None:
        return []
    return data.split(",")
```

### 5. Adicionar Teste de Regressão

**SEMPRE adicionar teste que:**
- Falha antes do fix
- Passa depois do fix
- Previne regressão futura

```python
# tests/test_bugfix_123.py
import pytest
from deepbridge.core import process_data

def test_process_data_handles_none():
    """Regression test for issue #123: process_data should handle None."""
    result = process_data(None)
    assert result == []

def test_process_data_normal_case():
    """Ensure fix doesn't break normal case."""
    result = process_data("a,b,c")
    assert result == ["a", "b", "c"]
```

### 6. Rodar Testes

```bash
# Rodar suite completa
pytest

# Rodar teste específico
pytest tests/test_bugfix_123.py -v

# Verificar cobertura
pytest --cov=deepbridge --cov-report=html
```

**Critérios de aceitação:**
- ✅ Todos os testes passam
- ✅ Novo teste de regressão incluído
- ✅ Cobertura mantida ou aumentada
- ✅ Linting passa (`ruff check .`)

### 7. Commit e Push

```bash
# Adicionar mudanças
git add .

# Commit seguindo conventional commits
git commit -m "fix: handle None input in process_data (fixes #123)"

# Push
git push origin fix/issue-123-import-error
```

### 8. Criar Pull Request

Use o comando `gh pr create` com título e corpo descritivos incluindo:
- Summary do fix
- Mudanças realizadas
- Testes executados
- Tipo de mudança

### 9. Code Review e Merge

**Antes de fazer merge:**
- ✅ CI passa (testes, linting, type checking)
- ✅ Code review aprovado
- ✅ Conflitos resolvidos
- ✅ Changelog atualizado (se necessário)

### 10. Atualizar Issue

Na issue original:
```markdown
✅ **Fixed in PR #456**

Will be available in next release (2.0.1).

**Workaround until then:**
[se aplicável]
```

---

## 🚨 Workflow de Hotfix (Bug Crítico)

Para bugs **críticos** que exigem release imediato:

### 1. Avaliar se é Realmente Crítico

**Critérios para hotfix:**
- ✅ Quebra funcionalidade essencial
- ✅ Vulnerabilidade de segurança
- ✅ Perda de dados
- ✅ Impossibilita uso do sistema
- ❌ Bug menor (pode esperar próxima release)

### 2. Criar Hotfix Branch

```bash
# Branch direto da tag de produção
git checkout -b hotfix/2.0.1 v2.0.0

# Ou da main se já estável
git checkout -b hotfix/2.0.1 master
```

### 3. Implementar Fix (Processo Acelerado)

**Mesmos passos do bugfix regular, mas:**
- ⚡ Prioridade máxima
- 🎯 Fix mínimo e conservador
- ✅ Testes essenciais (não suite completa se urgente)
- 📝 Documentar razão da urgência

### 4. Bump de Versão

```bash
# Atualizar versão em todos os lugares
# deepbridge/setup.py
version="2.0.1"

# deepbridge/__init__.py
__version__ = "2.0.1"

# Commit
git commit -m "chore: bump version to 2.0.1 (hotfix)"
```

### 5. Release Imediato

```bash
# Tag
git tag -a v2.0.1 -m "Hotfix: critical bugfix for [issue]"

# Push
git push origin hotfix/2.0.1 --tags

# Build e publish (ver WORKFLOW_RELEASE.md)
python -m build
twine upload dist/*
```

### 6. Comunicar Usuários

Criar GitHub Release com notas explicando o problema crítico e a solução.

### 7. Merge de Volta para Main

```bash
# Merge hotfix de volta para desenvolvimento
git checkout master
git merge hotfix/2.0.1
git push origin master

# Deletar branch
git branch -d hotfix/2.0.1
git push origin --delete hotfix/2.0.1
```

---

## 📦 Patch Release Process

Para releases regulares de correções (não emergenciais):

### 1. Agrupar Bugfixes

**Quando lançar patch release:**
- Acumulou 3-5 bugfixes importantes
- Passou 1-2 semanas desde último release
- Usuários pedindo fix específico

### 2. Preparar Release

```bash
# Branch de release
git checkout -b release/2.0.1 master

# Atualizar CHANGELOG.md com os fixes
# Bump versão
# Commit
git commit -m "chore: prepare release 2.0.1"
```

### 3. Testar Release Candidate

```bash
# Build
python -m build

# Test install em ambiente limpo
python -m venv test_release
source test_release/bin/activate
pip install dist/deepbridge-2.0.1-*.whl

# Rodar smoke tests
python -c "import deepbridge; print(deepbridge.__version__)"
pytest tests/smoke/
```

### 4. Lançar Release

```bash
# Tag e push
git tag -a v2.0.1 -m "Release v2.0.1"
git push origin release/2.0.1 --tags

# Publish
twine upload dist/*

# Merge para master
git checkout master
git merge release/2.0.1
git push origin master
```

### 5. Criar GitHub Release

Use `gh release create` com notas de release detalhadas.

---

## 📝 Templates

### Template de Commit Message (Bugfix)

```
fix: [descrição curta] (fixes #issue)

[Descrição detalhada do problema]
[Descrição detalhada da solução]
[Impactos e considerações]

Closes #issue
```

### Template de Commit Message (Hotfix)

```
fix(critical): [descrição curta] (fixes #issue)

⚠️ HOTFIX: [Razão da urgência]

[Descrição do problema crítico]
[Descrição da solução]
[Passos de verificação]

Closes #issue
```

---

## 🔍 Debugging Tips

### Reproduzir Bugs Reportados

```bash
# 1. Isolar ambiente
python -m venv debug_env && source debug_env/bin/activate

# 2. Instalar versão exata
pip install deepbridge==2.0.0

# 3. Copiar código do report
# 4. Adicionar prints e breakpoints
import pdb; pdb.set_trace()

# 5. Rodar com verbose
python -v reproduce_bug.py
```

### Logs Detalhados

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Seu código aqui
```

### Profiling

```bash
# CPU profiling
python -m cProfile -o profile.stats buggy_code.py
python -m pstats profile.stats

# Memory profiling
pip install memory_profiler
python -m memory_profiler buggy_code.py
```

---

## ✅ Checklist de Verificação

### Antes de Fazer Commit

- [ ] Bug reproduzido localmente
- [ ] Fix implementado e testado
- [ ] Teste de regressão adicionado
- [ ] Todos os testes passam
- [ ] Linting passa
- [ ] Type checking passa (mypy)
- [ ] Código revisado (self-review)
- [ ] Comentários adicionados se necessário
- [ ] Issue referenciada no commit

### Antes de Fazer Merge

- [ ] CI verde
- [ ] Code review aprovado
- [ ] Sem conflitos
- [ ] Changelog atualizado (se patch release)
- [ ] Documentação atualizada (se necessário)

### Antes de Lançar Patch Release

- [ ] Todos os bugfixes incluídos testados
- [ ] Versão atualizada em todos os lugares
- [ ] CHANGELOG atualizado
- [ ] Tag criada
- [ ] Build testado em ambiente limpo
- [ ] Release notes preparadas

---

## 📊 Métricas

Acompanhar:
- Tempo médio para resolver bugs
- Taxa de regressão (bugs reabertos)
- Número de hotfixes vs. patches regulares
- Cobertura de testes de regressão

---

**Última atualização:** 2025-02-16

Para mais detalhes sobre releases, consulte `WORKFLOW_RELEASE.md`.
