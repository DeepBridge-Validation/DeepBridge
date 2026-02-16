# Workflow de Bugfix - DeepBridge v2.0

**Última atualização:** 2026-02-16

Este documento descreve o processo completo para corrigir bugs no ecossistema DeepBridge, desde a identificação até a release do patch.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Classificação de Bugs](#classificação-de-bugs)
3. [Workflow Padrão de Bugfix](#workflow-padrão-de-bugfix)
4. [Workflow de Hotfix (Bugs Críticos)](#workflow-de-hotfix-bugs-críticos)
5. [Processo de Patch Release](#processo-de-patch-release)
6. [Templates de Commit](#templates-de-commit)
7. [Checklist de Bugfix](#checklist-de-bugfix)
8. [Exemplos Práticos](#exemplos-práticos)

---

## Visão Geral

### Princípios

1. **Reprodutibilidade:** Todo bug deve ser reproduzível antes de ser corrigido
2. **Testes:** Toda correção deve incluir testes que falham antes e passam depois
3. **Documentação:** Mudanças devem ser documentadas no CHANGELOG
4. **Rastreabilidade:** Commits devem referenciar a issue do bug
5. **Velocidade:** Bugs críticos devem ser corrigidos em < 24h

### SLA (Service Level Agreement)

| Prioridade | Tempo de Resposta | Tempo de Resolução | Processo |
|------------|-------------------|---------------------|----------|
| 🔴 **Crítico** | < 2 horas | < 24 horas | Hotfix |
| 🟠 **Alto** | < 8 horas | < 3 dias | Standard |
| 🟡 **Médio** | < 24 horas | < 1 semana | Standard |
| 🟢 **Baixo** | < 48 horas | Next release | Standard |

---

## Classificação de Bugs

### 🔴 Crítico (Priority: Critical)

**Características:**
- Bloqueia uso do sistema
- Perda de dados
- Vulnerabilidade de segurança
- Quebra de API pública sem aviso

**Exemplos:**
- `ImportError` que impede uso do pacote
- Crash ao inicializar
- Vazamento de memória crítico
- SQL injection ou XSS

**Ação:** Hotfix imediato

---

### 🟠 Alto (Priority: High)

**Características:**
- Funcionalidade principal não funciona
- Workaround existe mas é complexo
- Afeta muitos usuários

**Exemplos:**
- Método principal retorna resultado incorreto
- Performance drasticamente degradada
- Incompatibilidade com versão comum de dependência

**Ação:** Bugfix prioritário no próximo patch

---

### 🟡 Médio (Priority: Medium)

**Características:**
- Funcionalidade secundária não funciona
- Workaround simples existe
- Afeta poucos usuários

**Exemplos:**
- Mensagem de erro confusa
- Parâmetro opcional não funciona
- Documentação desatualizada

**Ação:** Bugfix no próximo minor/patch

---

### 🟢 Baixo (Priority: Low)

**Características:**
- Problema cosmético
- Não afeta funcionalidade
- Impacto mínimo

**Exemplos:**
- Typo em comentário
- Warning desnecessário
- Melhoria de mensagem de log

**Ação:** Pode esperar próximo release

---

## Workflow Padrão de Bugfix

### 1. Triagem e Reprodução

**1.1 Confirmar a Issue**
- Ler a issue completamente
- Verificar se tem informações suficientes
- Pedir informações adicionais se necessário

**1.2 Reproduzir o Bug**
```bash
# Criar ambiente isolado
python -m venv venv_bugfix
source venv_bugfix/bin/activate

# Instalar versão afetada
pip install deepbridge==X.Y.Z

# Tentar reproduzir com código do usuário
python test_bug.py
```

**1.3 Criar Teste que Falha**
```python
# tests/test_bugfix_issue_123.py
import pytest
from deepbridge import ...

def test_bug_issue_123():
    """
    Reproduz bug reportado em #123
    Expected: X
    Actual: Y (antes do fix)
    """
    # Código que demonstra o bug
    result = function_with_bug()
    assert result == expected_result  # Falha antes do fix
```

---

### 2. Criar Branch de Fix

```bash
# Nomenclatura: fix/issue-{number}-{description}
git checkout -b fix/issue-123-import-error

# Exemplo específico
git checkout -b fix/issue-123-distillation-import-error
```

**Convenções de nomenclatura:**
- `fix/issue-{n}-{short-desc}` - Bug com issue
- `fix/{short-desc}` - Bug sem issue (descoberto internamente)
- `hotfix/{short-desc}` - Bug crítico

---

### 3. Implementar Correção

**3.1 Localizar a Causa Raiz**
```bash
# Usar debugger
python -m pdb script_with_bug.py

# Adicionar logs temporários
import logging
logging.basicConfig(level=logging.DEBUG)
```

**3.2 Implementar Fix**
- Fazer a menor mudança possível que corrija o bug
- Evitar refactorings grandes
- Manter compatibilidade retroativa quando possível

**3.3 Verificar que Teste Agora Passa**
```bash
# Rodar teste específico
pytest tests/test_bugfix_issue_123.py -v

# Rodar suite completa para evitar regressões
pytest tests/ -v
```

---

### 4. Documentar a Correção

**4.1 Atualizar CHANGELOG.md**
```markdown
## [2.0.1] - 2026-02-16

### Fixed
- Fixed ImportError when importing KnowledgeDistiller (#123)
- Fixed memory leak in training loop (#124)
```

**4.2 Adicionar Docstring se Relevante**
```python
def fixed_function():
    """
    Function description.

    Note:
        Fixed in v2.0.1: Correctly handles edge case X (#123)
    """
```

---

### 5. Criar Pull Request

**5.1 Commit com Mensagem Descritiva**
```bash
git add .
git commit -m "fix: resolve ImportError in distillation module

- Add missing __init__.py import
- Add test to prevent regression
- Update CHANGELOG.md

Fixes #123"
```

**5.2 Push e Abrir PR**
```bash
git push origin fix/issue-123-import-error

# Abrir PR via GitHub CLI
gh pr create \
  --title "fix: resolve ImportError in distillation module (#123)" \
  --body "$(cat <<'EOF'
## Summary
Fixes #123 - ImportError when importing KnowledgeDistiller

## Changes
- Added missing import in `deepbridge/distillation/__init__.py`
- Added regression test in `tests/test_distillation_imports.py`
- Updated CHANGELOG.md

## Testing
- [x] Added test that reproduces the bug
- [x] Test passes after fix
- [x] All existing tests pass
- [x] Manual testing performed

## Breaking Changes
None - backward compatible

---
🤖 Generated with Claude Code
EOF
)"
```

---

### 6. Review e Merge

**6.1 Code Review**
- Aguardar aprovação de maintainer
- Responder comentários
- Fazer ajustes se necessário

**6.2 CI/CD Checks**
- Verificar que todos os testes passam
- Verificar cobertura de código
- Verificar linting

**6.3 Merge**
```bash
# Usar squash merge para manter histórico limpo
gh pr merge --squash --delete-branch
```

---

## Workflow de Hotfix (Bugs Críticos)

Para bugs **críticos** que precisam ser corrigidos imediatamente:

### 1. Notificação Imediata

```bash
# Abrir issue com tag [CRITICAL]
gh issue create \
  --title "[CRITICAL] Production ImportError blocking all users" \
  --label "bug,priority:critical" \
  --body "..."

# Notificar equipe (Discord, Slack, email)
```

---

### 2. Branch Direto de Main

```bash
# Criar branch de hotfix
git checkout main
git pull origin main
git checkout -b hotfix/critical-import-error
```

---

### 3. Fix Rápido mas Testado

```bash
# Implementar fix
# Escrever teste mínimo
pytest tests/test_hotfix.py -v

# Rodar suite completa
pytest tests/ -v
```

---

### 4. PR Expedito

```bash
# Commit e push
git add .
git commit -m "hotfix: resolve critical ImportError blocking users

CRITICAL: This fix addresses a production issue affecting all users.

- Fix: Added missing import
- Test: Regression test added
- Impact: All users unable to import module

Fixes #999"

git push origin hotfix/critical-import-error

# PR com label priority:critical
gh pr create --label "priority:critical" --title "..." --body "..."
```

---

### 5. Release Imediato

```bash
# Após merge, release imediato
# Ver seção "Processo de Patch Release"
```

---

## Processo de Patch Release

### 1. Preparar Release

**1.1 Verificar Mudanças**
```bash
# Ver commits desde última release
git log v2.0.0..HEAD --oneline

# Ver CHANGELOG
cat CHANGELOG.md
```

**1.2 Atualizar Versão**
```bash
# Atualizar version em setup.py ou pyproject.toml
# Versão segue Semantic Versioning (MAJOR.MINOR.PATCH)

# Exemplo: 2.0.0 → 2.0.1 (bugfix)
# __version__ = "2.0.1"
```

**1.3 Atualizar CHANGELOG**
```markdown
## [2.0.1] - 2026-02-16

### Fixed
- Fixed ImportError when importing KnowledgeDistiller (#123)
- Fixed memory leak in training loop (#124)
- Fixed incorrect parameter validation (#125)

### Security
- Fixed potential XSS in report generation (#126)
```

---

### 2. Criar Tag e Release

**2.1 Commit de Release**
```bash
git add setup.py CHANGELOG.md
git commit -m "chore: release v2.0.1

- Bump version to 2.0.1
- Update CHANGELOG with bugfixes

Release notes:
- Fix: ImportError in distillation (#123)
- Fix: Memory leak in training (#124)
- Fix: Parameter validation (#125)
- Security: XSS in reports (#126)"
```

**2.2 Criar Tag**
```bash
# Tag anotada com mensagem
git tag -a v2.0.1 -m "Release v2.0.1 - Critical bugfixes

Fixes:
- ImportError in distillation module (#123)
- Memory leak in training loop (#124)
- Parameter validation issue (#125)
- XSS vulnerability in reports (#126)"

# Push tag
git push origin v2.0.1
```

**2.3 Criar GitHub Release**
```bash
gh release create v2.0.1 \
  --title "v2.0.1 - Critical Bugfixes" \
  --notes "$(cat <<'EOF'
## 🐛 Bugfixes

This patch release addresses several critical issues:

### Fixed
- **#123** - ImportError when importing KnowledgeDistiller
- **#124** - Memory leak in training loop
- **#125** - Incorrect parameter validation

### Security
- **#126** - Fixed potential XSS vulnerability in report generation

## 📦 Installation

```bash
pip install --upgrade deepbridge
```

## 🔄 Migration

No breaking changes - drop-in replacement for 2.0.0.

---

**Full Changelog:** https://github.com/guhaase/DeepBridge/compare/v2.0.0...v2.0.1
EOF
)"
```

---

### 3. Publicar no PyPI

**3.1 Build**
```bash
# Limpar builds anteriores
rm -rf dist/ build/ *.egg-info

# Build novo pacote
python -m build
```

**3.2 Verificar Build**
```bash
# Listar arquivos gerados
ls -lh dist/

# Verificar conteúdo
tar -tzf dist/deepbridge-2.0.1.tar.gz | head -20
```

**3.3 Publicar**
```bash
# Upload para PyPI
python -m twine upload dist/*

# Verificar
pip install --upgrade deepbridge
python -c "import deepbridge; print(deepbridge.__version__)"
# Deve mostrar: 2.0.1
```

---

### 4. Comunicação

**4.1 Anunciar no GitHub**
- Release notes já criadas no passo 2.3

**4.2 Anunciar em Canais**
- Twitter/X
- Discord/Slack
- Mailing list (se houver)

**Template de anúncio:**
```
🐛 DeepBridge v2.0.1 Released!

This patch release fixes several critical bugs:
- ImportError in distillation module
- Memory leak in training
- XSS vulnerability in reports

Upgrade now:
pip install --upgrade deepbridge

Full notes: https://github.com/guhaase/DeepBridge/releases/tag/v2.0.1
```

---

## Templates de Commit

### Bug Fix Padrão
```
fix: [short description]

- Detailed explanation of the bug
- What was causing it
- How it was fixed

Fixes #[issue-number]
```

### Hotfix Crítico
```
hotfix: [short description]

CRITICAL: [Why this is critical]

- Fix: [What was fixed]
- Test: [Test added]
- Impact: [Who is affected]

Fixes #[issue-number]
```

### Bugfix com Breaking Change (evitar!)
```
fix!: [short description]

BREAKING CHANGE: [What breaks]

- Why this breaking change is necessary
- Migration path for users
- Deprecation warnings added

Fixes #[issue-number]
```

---

## Checklist de Bugfix

### Antes de Começar
- [ ] Bug reproduzido localmente
- [ ] Prioridade classificada corretamente
- [ ] Issue criada com label apropriada
- [ ] Teste que falha criado

### Durante Desenvolvimento
- [ ] Causa raiz identificada
- [ ] Fix implementado (mínimo necessário)
- [ ] Teste agora passa
- [ ] Todos os testes existentes passam
- [ ] Nenhuma regressão introduzida
- [ ] Código revisado (self-review)

### Antes do PR
- [ ] CHANGELOG.md atualizado
- [ ] Commit message segue template
- [ ] Documentação atualizada (se relevante)
- [ ] Compatibilidade retroativa mantida (se possível)

### No PR
- [ ] Título descritivo
- [ ] Descrição completa com contexto
- [ ] Referência à issue (`Fixes #123`)
- [ ] Labels apropriadas
- [ ] CI/CD checks passando

### Para Release
- [ ] Versão incrementada (PATCH)
- [ ] CHANGELOG atualizado com data
- [ ] Tag criada
- [ ] Release notes escritas
- [ ] PyPI publicado
- [ ] Comunicação feita

---

## Exemplos Práticos

### Exemplo 1: Bug de Import (Prioridade: Alta)

**Issue:** #123 - `ImportError: cannot import name 'KnowledgeDistiller'`

**Workflow:**
```bash
# 1. Reproduzir
python -c "from deepbridge.distillation import KnowledgeDistiller"
# ImportError!

# 2. Criar branch
git checkout -b fix/issue-123-distillation-import

# 3. Criar teste
cat > tests/test_issue_123.py <<EOF
def test_import_knowledge_distiller():
    from deepbridge.distillation import KnowledgeDistiller
    assert KnowledgeDistiller is not None
EOF

# 4. Rodar teste (deve falhar)
pytest tests/test_issue_123.py
# FAILED

# 5. Identificar problema
# Falta importar em __init__.py

# 6. Fix
echo "from .distiller import KnowledgeDistiller" >> deepbridge/distillation/__init__.py

# 7. Testar novamente
pytest tests/test_issue_123.py
# PASSED

# 8. Commit e PR
git add .
git commit -m "fix: add missing KnowledgeDistiller import

- Added import in deepbridge/distillation/__init__.py
- Added regression test

Fixes #123"

git push origin fix/issue-123-distillation-import
gh pr create --title "fix: add missing KnowledgeDistiller import (#123)"
```

---

### Exemplo 2: Memory Leak (Prioridade: Crítica)

**Issue:** #124 - Memory usage grows unbounded during training

**Workflow:**
```bash
# 1. Reproduzir com profiler
python -m memory_profiler train_script.py
# Confirma: memória cresce continuamente

# 2. Hotfix branch
git checkout main
git checkout -b hotfix/memory-leak-training

# 3. Identificar causa
# Tensors não liberados após backward()

# 4. Fix
# Adicionar .detach() ou torch.no_grad()

# 5. Verificar fix
python -m memory_profiler train_script.py
# Memória agora estável

# 6. Commit e release IMEDIATO
git add .
git commit -m "hotfix: fix memory leak in training loop

CRITICAL: Memory usage was growing unbounded.

- Fix: Properly detach tensors after backward()
- Test: Memory usage profiling added
- Impact: All users training models

Fixes #124"

# 7. PR expedito
gh pr create --label "priority:critical"

# 8. Após merge: release 2.0.1 imediatamente
```

---

## Recursos Adicionais

- **Issue Templates:** `.github/ISSUE_TEMPLATE/bug_report.md`
- **PR Template:** `.github/PULL_REQUEST_TEMPLATE.md`
- **CI/CD:** `.github/workflows/`
- **Contributing Guide:** `CONTRIBUTING.md`
- **Plano de Contingência:** `desenvolvimento/refatoracao/PLANO_CONTINGENCIA.md`

---

**Dúvidas?** Abra uma issue com label `question` ou consulte o FAQ: `desenvolvimento/refatoracao/FAQ_V2.md`
