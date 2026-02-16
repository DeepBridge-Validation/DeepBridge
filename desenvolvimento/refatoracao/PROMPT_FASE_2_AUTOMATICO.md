# PROMPT PARA EXECUÇÃO AUTOMÁTICA - FASE 2: Migração de Código

**IMPORTANTE:** Este prompt foi projetado para execução 100% automática pelo Claude Code.

---

## 🎯 OBJETIVO

Migrar o código de distillation e synthetic dos backups da Fase 1 para os novos repositórios:
- `deepbridge-distillation`
- `deepbridge-synthetic`

Configurar estrutura completa com:
- Código migrado
- Imports ajustados
- `pyproject.toml` configurado
- README.md criado
- CI/CD configurado
- Commits e push realizados

---

## 📋 TAREFAS A EXECUTAR

Execute as seguintes tarefas em ordem, marcando cada uma como concluída conforme avança:

### Tarefa 1: Verificar pré-requisitos
- Verificar que backup da Fase 1 existe em `/tmp/deepbridge-migration/`
- Criar diretório de trabalho `/home/guhaase/projetos/deepbridge-v2`

### Tarefa 2: Clonar repositórios
- Clonar `https://github.com/DeepBridge-Validation/deepbridge-distillation.git`
- Clonar `https://github.com/DeepBridge-Validation/deepbridge-synthetic.git`

### Tarefa 3: Configurar deepbridge-distillation
- Criar estrutura de diretórios
- Copiar código do backup
- Ajustar imports (deepbridge.distillation → deepbridge_distillation)
- Criar `__init__.py`
- Criar `pyproject.toml` com dependência `deepbridge>=2.0.0-alpha.1`
- Criar `README.md`
- Configurar CI/CD (`.github/workflows/tests.yml`)
- Commit e push

### Tarefa 4: Configurar deepbridge-synthetic
- Criar estrutura de diretórios
- Copiar código do backup
- Ajustar imports (deepbridge.synthetic → deepbridge_synthetic)
- Criar `__init__.py`
- Criar `pyproject.toml` (sem dependência de deepbridge)
- Criar `README.md`
- Configurar CI/CD
- Commit e push

### Tarefa 5: Testar instalações
- Testar instalação de deepbridge-distillation
- Testar instalação de deepbridge-synthetic
- Verificar imports funcionando

---

## ⚙️ EXECUÇÃO

Por favor, execute todas as tarefas acima de forma **100% automática**.

Use:
- `Bash` para comandos git, mkdir, cp, find, sed
- `Write` para criar novos arquivos
- `TodoWrite` para rastrear progresso

Siga exatamente os comandos especificados em FASE_2_MIGRACAO_CODIGO.md.

**IMPORTANTE:**
- Substitua todos os imports corretamente
- Garanta que pyproject.toml do distillation depende de deepbridge>=2.0.0-alpha.1
- Garanta que pyproject.toml do synthetic NÃO depende de deepbridge
- Faça commits descritivos
- Push para branch `main` de cada novo repo

---

## 🔍 VERIFICAÇÃO FINAL

Ao finalizar, confirme que:
1. ✅ Ambos os repos foram clonados
2. ✅ Código foi copiado do backup
3. ✅ Imports foram ajustados
4. ✅ pyproject.toml criados corretamente
5. ✅ README.md criados
6. ✅ CI/CD configurado
7. ✅ Commits e push realizados
8. ✅ Testes de import passando

---

## 📝 REFERÊNCIA

Para detalhes completos, consulte:
`/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/FASE_2_MIGRACAO_CODIGO.md`

---

**EXECUTAR AGORA DE FORMA 100% AUTOMÁTICA**
