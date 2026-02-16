# Guia Rápido: Prompts para Execução Automática

**🎯 Objetivo:** Executar Fases 2-6 da migração DeepBridge v2.0 de forma automática via Claude Code.

---

## ⚡ Início Rápido

### Executar Próxima Fase (Fase 2)

```bash
# 1. Visualizar o prompt
cat desenvolvimento/refatoracao/PROMPT_FASE_2_AUTOMATICO.md

# 2. Copiar todo o conteúdo

# 3. Colar no Claude Code e enviar
```

**Pronto!** O Claude Code executará automaticamente toda a Fase 2.

---

## 📚 Arquivos Disponíveis

| Prompt | Automação | Para que serve |
|--------|-----------|----------------|
| `PROMPT_FASE_2_AUTOMATICO.md` | 🟢 100% | Migrar código para novos repos |
| `PROMPT_FASE_3_AUTOMATICO.md` | 🟡 85% | Configurar e executar testes |
| `PROMPT_FASE_4_AUTOMATICO.md` | 🟢 100% | Criar documentação e exemplos |
| `PROMPT_FASE_5_AUTOMATICO.md` | 🔴 50% | Release PyPI (requer tokens) |
| `PROMPT_FASE_6_AUTOMATICO.md` | 🟡 70% | Configurar suporte pós-launch |

**Índice completo:** `INDICE_PROMPTS_AUTOMATICOS.md`

---

## 🎓 Níveis de Automação

### 🟢 100% Automático (Fases 2 e 4)
- Não requer intervenção
- Execute e aguarde
- Tudo será feito automaticamente

### 🟡 85% Automático (Fase 3)
- Maioria automática
- Pode precisar corrigir falhas de testes
- Claude tentará corrigir automaticamente primeiro

### 🔴 50% Automático (Fase 5)
- Preparação automática
- Publicação PyPI requer tokens manuais
- Escolha modo híbrido ou só preparação

---

## 🔄 Ordem de Execução

```
✅ Fase 1: Concluída (2026-02-16)
    ↓
📦 Fase 2: PROMPT_FASE_2_AUTOMATICO.md  ← PRÓXIMA
    ↓
🧪 Fase 3: PROMPT_FASE_3_AUTOMATICO.md
    ↓
📝 Fase 4: PROMPT_FASE_4_AUTOMATICO.md
    ↓
🚀 Fase 5: PROMPT_FASE_5_AUTOMATICO.md  (requer tokens)
    ↓
🛠️ Fase 6: PROMPT_FASE_6_AUTOMATICO.md  (configuração)
```

---

## 💡 Dicas

1. **Execute uma fase por vez** - Não pule fases
2. **Verifique checklist** ao final de cada fase
3. **Leia avisos** no prompt (ex: "requer token")
4. **Confirme git status** após cada fase
5. **Faça backup** antes de começar (já feito na Fase 1)

---

## 🆘 Troubleshooting

### Erro: "Backup não encontrado"
**Solução:** Execute primeiro `FASE_1_PREPARACAO.md`

### Erro: "Git remote not found"
**Solução:** Verifique que criou os repos no GitHub

### Erro: "PyPI token invalid"
**Solução:** Configure tokens antes da Fase 5:
```bash
poetry config pypi-token.testpypi pypi-YOUR_TOKEN
```

---

## 📞 Suporte

Para dúvidas sobre os prompts:
- Consulte: `INDICE_PROMPTS_AUTOMATICOS.md`
- Consulte arquivo original: `FASE_X_*.md`
- Abra issue: https://github.com/DeepBridge-Validation/DeepBridge/issues

---

**Versão:** 1.0
**Data:** 2026-02-16
**Status:** ✅ Pronto para uso

**Próxima ação:** Execute Fase 2
