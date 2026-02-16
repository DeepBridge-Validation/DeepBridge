# Instruções para Criar Labels no GitHub

Este documento explica como criar os labels necessários nos repositórios DeepBridge.

---

## 🎯 Objetivo

Criar labels padronizados nos 3 repositórios:
- `guhaase/deepbridge`
- `guhaase/deepbridge-distillation`
- `guhaase/deepbridge-synthetic`

---

## 📋 Labels a Criar

### Labels de Tipo
- **bug** (vermelho) - Something isn't working
- **enhancement** (verde) - New feature or request
- **documentation** (azul) - Improvements or additions to documentation
- **question** (amarelo) - Further information is requested
- **migration** (roxo) - Related to migration from v1.x to v2.x

### Labels de Prioridade
- **priority: critical** (vermelho escuro) - Critical priority - needs immediate attention
- **priority: high** (laranja) - High priority
- **priority: medium** (amarelo) - Medium priority
- **priority: low** (verde claro) - Low priority

---

## ⚙️ Método 1: Script Automático (Recomendado)

### Pré-requisitos
1. Ter o GitHub CLI instalado:
   ```bash
   # Ubuntu/Debian
   sudo apt install gh

   # macOS
   brew install gh

   # Windows
   winget install GitHub.cli
   ```

2. Autenticar no GitHub:
   ```bash
   gh auth login
   ```

### Execução

```bash
# Do diretório desenvolvimento/
./scripts/create_github_labels.sh
```

O script criará todos os labels nos 3 repositórios automaticamente.

---

## ⚙️ Método 2: Manualmente no GitHub

Se preferir criar manualmente:

1. Acesse cada repositório no GitHub
2. Vá em **Settings** → **Labels**
3. Clique em **New label** para cada um:

### Repositório: deepbridge

| Nome | Cor | Descrição |
|------|-----|-----------|
| bug | `#d73a4a` | Something isn't working |
| enhancement | `#0e8a16` | New feature or request |
| documentation | `#0075ca` | Improvements or additions to documentation |
| question | `#d876e3` | Further information is requested |
| priority: critical | `#b60205` | Critical priority - needs immediate attention |
| priority: high | `#d93f0b` | High priority |
| priority: medium | `#fbca04` | Medium priority |
| priority: low | `#c2e0c6` | Low priority |
| migration | `#5319e7` | Related to migration from v1.x to v2.x |

### Repositório: deepbridge-distillation

Repetir os mesmos labels acima.

### Repositório: deepbridge-synthetic

Repetir os mesmos labels acima.

---

## ✅ Verificação

Após criar os labels, verifique:

1. Acesse cada repositório no GitHub
2. Vá em **Settings** → **Labels**
3. Confirme que todos os 9 labels estão presentes
4. Confirme que as cores estão corretas

---

## 🔍 Uso dos Labels

### Ao Criar Issue

Os templates de issue automaticamente sugerem labels apropriados:
- Bug reports → `bug`
- Feature requests → `enhancement`
- Questions → `question`

### Priorização

Adicione labels de prioridade conforme a urgência:
- `priority: critical` - Bugs que quebram funcionalidade essencial
- `priority: high` - Problemas importantes mas não bloqueantes
- `priority: medium` - Melhorias e bugs menores
- `priority: low` - Nice to have

### Migração

Para issues relacionadas à migração v1.x → v2.x, adicione:
- `migration`

---

## 📝 Notas

- Labels podem ser editados depois se necessário
- A descrição ajuda usuários a entender quando usar cada label
- Cores consistentes facilitam identificação visual
- Labels são independentes entre repositórios (precisam ser criados em cada um)

---

## 🆘 Troubleshooting

### "gh: command not found"
Instale o GitHub CLI seguindo instruções em: https://cli.github.com/

### "authentication required"
Execute `gh auth login` e siga as instruções.

### "HTTP 404: Not Found"
Verifique se você tem permissão de admin nos repositórios.

### Labels já existem
O script usa `--force` que atualiza labels existentes. Seguro executar múltiplas vezes.

---

**Próximo passo:** Após criar os labels, o sistema de suporte está 100% pronto para receber issues!
