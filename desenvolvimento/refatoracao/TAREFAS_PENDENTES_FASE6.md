# Tarefas Pendentes - Fase 6 (Ação Manual Requerida)

**Data:** 2026-02-16
**Status:** ⚠️ AGUARDANDO AÇÃO MANUAL

---

## 📋 Resumo

A **Fase 6** foi executada com sucesso de forma automática, mas **uma tarefa crítica** requer sua ação manual devido à necessidade de autenticação com credenciais do GitHub.

---

## ⚠️ TAREFA PENDENTE: Criar Labels no GitHub

### O que fazer:

Executar o script que cria labels padronizados nos 3 repositórios:
- `DeepBridge-Validation/DeepBridge`
- `DeepBridge-Validation/deepbridge-distillation`
- `DeepBridge-Validation/deepbridge-synthetic`

### Por que é necessário:

Os labels são usados para:
- Categorizar issues (bug, enhancement, documentation, question, migration)
- Priorizar trabalho (priority: critical, high, medium, low)
- Facilitar triagem e organização do projeto

---

## 🚀 Como Executar (3 passos)

### Passo 1: Instalar GitHub CLI (se ainda não tiver)

```bash
# Ubuntu/Debian
sudo apt install gh

# macOS
brew install gh

# Windows
winget install GitHub.cli
```

**Verificar instalação:**
```bash
gh --version
# Deve mostrar: gh version X.X.X
```

---

### Passo 2: Autenticar no GitHub

```bash
gh auth login
```

Siga as instruções:
1. Escolha: **GitHub.com**
2. Escolha: **HTTPS**
3. Escolha: **Login with a web browser**
4. Copie o código que aparece
5. Pressione Enter para abrir o navegador
6. Cole o código e autorize

**Verificar autenticação:**
```bash
gh auth status
# Deve mostrar: ✓ Logged in to github.com as SEU_USERNAME
```

---

### Passo 3: Executar o Script

```bash
cd /home/guhaase/projetos/DeepBridge
./scripts/create_github_labels.sh
```

**O script irá:**
1. Verificar se `gh` está instalado e autenticado
2. Criar 9 labels em cada um dos 3 repositórios (27 labels no total)
3. Mostrar progresso em tempo real
4. Confirmar sucesso ao final

**Tempo estimado:** ~30 segundos

---

## 📊 Labels que Serão Criados

### Labels de Tipo (5)
- 🐛 **bug** (vermelho) - Something isn't working
- ✨ **enhancement** (verde) - New feature or request
- 📚 **documentation** (azul) - Improvements or additions to documentation
- ❓ **question** (amarelo) - Further information is requested
- 🔄 **migration** (roxo) - Related to migration from v1.x to v2.x

### Labels de Prioridade (4)
- 🚨 **priority: critical** (vermelho escuro) - Needs immediate attention
- ⚡ **priority: high** (laranja) - High priority
- ⏺️ **priority: medium** (amarelo) - Medium priority
- 🔵 **priority: low** (verde claro) - Low priority

---

## ✅ Verificação

Após executar o script, verifique os labels criados:

1. **DeepBridge:**
   https://github.com/DeepBridge-Validation/DeepBridge/labels

2. **deepbridge-distillation:**
   https://github.com/DeepBridge-Validation/deepbridge-distillation/labels

3. **deepbridge-synthetic:**
   https://github.com/DeepBridge-Validation/deepbridge-synthetic/labels

Cada repo deve ter **9 labels** com cores corretas.

---

## 🆘 Troubleshooting

### Erro: "gh: command not found"
**Solução:** Instale o GitHub CLI (Passo 1)

### Erro: "authentication required"
**Solução:** Execute `gh auth login` (Passo 2)

### Erro: "HTTP 404: Not Found"
**Solução:** Verifique se você tem permissão de admin nos repositórios

### Erro: "label already exists"
**Solução:** Não é um erro! O script usa `--force` e atualiza labels existentes

---

## 📝 Alternativa Manual

Se preferir criar manualmente via interface web do GitHub:

1. Acesse cada repositório
2. Vá em **Settings** → **Labels**
3. Clique em **New label**
4. Copie as informações da tabela acima

**Tempo estimado:** ~15 minutos (manual)

vs.

**Tempo com script:** ~30 segundos (automático)

---

## 🎯 Próximos Passos

Após criar os labels:

1. ✅ **Marcar como concluído** no checklist de PROMPT_FASE_6_AUTOMATICO.md
2. ✅ **Fase 6 estará 100% completa**
3. ✅ **Sistema de suporte totalmente operacional**
4. 🎉 **Pronto para receber issues da comunidade!**

---

## 📁 Arquivos Relacionados

- 📜 **Script:** `/home/guhaase/projetos/DeepBridge/scripts/create_github_labels.sh`
- 📖 **Instruções:** `/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/INSTRUCOES_LABELS_GITHUB.md`
- ✅ **Checklist:** `/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/PROMPT_FASE_6_AUTOMATICO.md`

---

## ⏱️ Tempo Total Estimado

- **Instalação gh CLI** (se necessário): 2-5 minutos
- **Autenticação GitHub**: 1-2 minutos
- **Execução do script**: 30 segundos
- **Verificação**: 1 minuto

**Total:** ~5-10 minutos (primeira vez)
**Total:** ~2 minutos (se gh já estiver instalado)

---

## 🎉 Conclusão

Esta é a **ÚNICA** tarefa pendente da Fase 6!

Tudo o mais foi executado automaticamente:
- ✅ Templates de issues criados (3 repos)
- ✅ FAQ criado e populado
- ✅ Scripts de métricas criados
- ✅ Workflow de bugfix documentado
- ✅ Plano de contingência criado
- ✅ Commits e push realizados

Após criar os labels, a **Fase 6 estará 100% completa!** 🎊

---

**Criado por:** Claude Code
**Última atualização:** 2026-02-16
