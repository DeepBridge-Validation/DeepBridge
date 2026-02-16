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

## ✅ CHECKLIST FINAL - VERIFICAR APÓS EXECUÇÃO

### Pré-requisitos
- [ ] Backup da Fase 1 existe em `/tmp/deepbridge-migration/`
- [ ] Diretório `/home/guhaase/projetos/deepbridge-v2/` criado
- [ ] Repositórios GitHub criados (deepbridge-distillation, deepbridge-synthetic)

### deepbridge-distillation
- [ ] Repositório clonado em `/home/guhaase/projetos/deepbridge-v2/deepbridge-distillation/`
- [ ] Estrutura de diretórios criada (deepbridge_distillation/, tests/, examples/, docs/)
- [ ] Código copiado de `/tmp/deepbridge-migration/distillation/`
- [ ] Testes copiados de `/tmp/deepbridge-migration/tests/test_distillation/`
- [ ] Imports ajustados (deepbridge.distillation → deepbridge_distillation)
- [ ] Arquivo `__init__.py` criado com versão 2.0.0-alpha.1
- [ ] Arquivo `pyproject.toml` criado com dependência `deepbridge>=2.0.0-alpha.1`
- [ ] Arquivo `README.md` criado com instalação e quick start
- [ ] CI/CD configurado (`.github/workflows/tests.yml`)
- [ ] Commit realizado com mensagem descritiva
- [ ] Push para branch `main` bem-sucedido
- [ ] Teste de import funcionando:
  - [ ] `import deepbridge_distillation`
  - [ ] `from deepbridge_distillation import AutoDistiller`
  - [ ] `import deepbridge` (deve funcionar como dependência)

### deepbridge-synthetic
- [ ] Repositório clonado em `/home/guhaase/projetos/deepbridge-v2/deepbridge-synthetic/`
- [ ] Estrutura de diretórios criada (deepbridge_synthetic/, tests/, examples/, docs/)
- [ ] Código copiado de `/tmp/deepbridge-migration/synthetic/`
- [ ] Testes copiados de `/tmp/deepbridge-migration/tests/test_synthetic/`
- [ ] Imports ajustados (deepbridge.synthetic → deepbridge_synthetic)
- [ ] Arquivo `__init__.py` criado com versão 2.0.0-alpha.1
- [ ] Arquivo `pyproject.toml` criado SEM dependência de deepbridge
- [ ] Arquivo `README.md` criado (destacando standalone)
- [ ] CI/CD configurado (`.github/workflows/tests.yml`)
- [ ] Commit realizado com mensagem descritiva
- [ ] Push para branch `main` bem-sucedido
- [ ] Teste de import funcionando:
  - [ ] `import deepbridge_synthetic`
  - [ ] `from deepbridge_synthetic import Synthesize`
  - [ ] `import deepbridge` NÃO deve funcionar (standalone)

### Verificações Finais
- [ ] Ambos os repos visíveis no GitHub
- [ ] Código-fonte migrado corretamente (sem erros de sintaxe)
- [ ] Imports todos ajustados (nenhum import do antigo deepbridge.distillation/synthetic)
- [ ] Dependências corretas em cada pyproject.toml
- [ ] README.md com instruções claras
- [ ] CI/CD configurado e pronto para usar

### Contagem de Arquivos
- [ ] deepbridge-distillation: ~22 arquivos Python
- [ ] deepbridge-synthetic: ~29 arquivos Python
- [ ] Total migrado: ~51 arquivos

---

**STATUS DA FASE 2:** ⬜ NÃO INICIADA | 🚧 EM ANDAMENTO | ✅ CONCLUÍDA

Marque como ✅ CONCLUÍDA apenas se TODOS os itens acima estiverem marcados.

---

**EXECUTAR AGORA DE FORMA 100% AUTOMÁTICA**
