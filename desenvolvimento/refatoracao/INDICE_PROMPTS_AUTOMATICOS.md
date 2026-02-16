# Índice de Prompts para Execução Automática

**Versão:** 1.0
**Data:** 2026-02-16
**Status:** ✅ PRONTO PARA USO

---

## 📚 Visão Geral

Este documento lista os prompts criados para execução automática das Fases 2-6 da migração DeepBridge v2.0.

Cada prompt foi projetado para ser enviado ao Claude Code para execução **máxima automação possível**.

---

## 📋 Lista de Prompts

### ✅ FASE 1: Preparação (CONCLUÍDA)
**Arquivo original:** `FASE_1_PREPARACAO.md`
**Status:** ✅ Executada com sucesso em 2026-02-16
**Prompt:** Não necessário (já executada)

**Resultado:**
- Branch `feat/split-repos-v2` criado
- Backup criado em `/tmp/deepbridge-migration/`
- Código de distillation/synthetic removido
- Versão atualizada para `2.0.0-alpha.1`
- Commit: `a7fcb0a`

---

### 📦 FASE 2: Migração de Código
**Arquivo original:** `FASE_2_MIGRACAO_CODIGO.md`
**Prompt executável:** `PROMPT_FASE_2_AUTOMATICO.md`
**Nível de automação:** 🟢 **100% automático**

**O que o prompt faz:**
1. Clona novos repositórios (deepbridge-distillation, deepbridge-synthetic)
2. Copia código do backup
3. Ajusta imports automaticamente
4. Cria pyproject.toml, README.md, CI/CD
5. Commit e push

**Pré-requisitos:**
- ✅ Fase 1 concluída
- ✅ Backup existe em `/tmp/deepbridge-migration/`
- ✅ Repositórios criados no GitHub

**Como executar:**
```bash
# Copie o conteúdo de PROMPT_FASE_2_AUTOMATICO.md e envie para Claude Code
# Ou simplesmente:
cat desenvolvimento/refatoracao/PROMPT_FASE_2_AUTOMATICO.md
# E cole no Claude Code
```

---

### 🧪 FASE 3: Migração de Testes
**Arquivo original:** `FASE_3_MIGRACAO_TESTES.md`
**Prompt executável:** `PROMPT_FASE_3_AUTOMATICO.md`
**Nível de automação:** 🟡 **85% automático**

**O que o prompt faz:**
1. Remove testes de distillation/synthetic do core
2. Cria conftest.py nos novos repos
3. Ajusta imports nos testes
4. Executa testes
5. Gera relatórios de coverage
6. Cria testes de integração

**Limitações:**
- ⚠️ Se testes falharem por **lógica de negócio**, pode precisar intervenção manual
- ✅ Falhas de import são corrigidas automaticamente
- ✅ Fixtures ausentes são criadas automaticamente

**Pré-requisitos:**
- ✅ Fase 2 concluída
- ✅ Código migrado para novos repos

**Como executar:**
```bash
cat desenvolvimento/refatoracao/PROMPT_FASE_3_AUTOMATICO.md
# Enviar para Claude Code
```

---

### 📝 FASE 4: Documentação e Exemplos
**Arquivo original:** `FASE_4_DOCUMENTACAO.md`
**Prompt executável:** `PROMPT_FASE_4_AUTOMATICO.md`
**Nível de automação:** 🟢 **100% automático**

**O que o prompt faz:**
1. Cria exemplos executáveis para cada repo
2. Cria CHANGELOG.md
3. Atualiza README.md com badges e links
4. Testa exemplos
5. Commit e push

**Pré-requisitos:**
- ✅ Fase 3 concluída
- ✅ Testes passando

**Como executar:**
```bash
cat desenvolvimento/refatoracao/PROMPT_FASE_4_AUTOMATICO.md
# Enviar para Claude Code
```

---

### 🚀 FASE 5: Release v2.0
**Arquivo original:** `FASE_5_RELEASE.md`
**Prompt executável:** `PROMPT_FASE_5_AUTOMATICO.md`
**Nível de automação:** 🔴 **50% automático** (requer credenciais PyPI)

**O que o prompt faz automaticamente:**
1. ✅ Atualiza versões (rc.1 → 2.0.0)
2. ✅ Cria tags
3. ✅ Build com poetry
4. ✅ Testa instalações
5. ✅ Cria release notes
6. ✅ Depreca v1.x

**O que REQUER intervenção manual:**
1. ⚠️ Configurar tokens PyPI (Test + Production)
2. ⚠️ Executar `poetry publish`
3. ⚠️ Validar no PyPI web

**Modos de execução:**
- **Opção A (Híbrido):** Automático com pausas para credenciais
- **Opção B (Preparação):** Só preparação automática, publicação manual
- **Opção C (Manual):** Apenas checklist

**Pré-requisitos:**
- ✅ Fase 4 concluída
- ✅ Documentação completa
- ⚠️ Tokens PyPI configurados

**Como executar:**
```bash
cat desenvolvimento/refatoracao/PROMPT_FASE_5_AUTOMATICO.md
# Enviar para Claude Code
# Escolher modo de execução (A, B ou C)
```

---

### 🛠️ FASE 6: Suporte Pós-Launch
**Arquivo original:** `FASE_6_SUPORTE.md`
**Prompt executável:** `PROMPT_FASE_6_AUTOMATICO.md`
**Nível de automação:** 🟡 **70% automático** (configuração)

**O que o prompt faz automaticamente:**
1. ✅ Cria templates de issues (.github/ISSUE_TEMPLATE/)
2. ✅ Cria FAQ dinâmico
3. ✅ Cria scripts de métricas
4. ✅ Documenta workflow de bugfix
5. ✅ Cria plano de contingência

**O que REQUER ação contínua manual:**
1. ⚠️ Monitorar e responder issues (diário)
2. ⚠️ Corrigir bugs e lançar patches
3. ⚠️ Atualizar FAQ com novos problemas
4. ⚠️ Executar scripts de métricas (semanal)

**Pré-requisitos:**
- ✅ Fase 5 concluída
- ✅ v2.0.0 publicado no PyPI

**Como executar:**
```bash
cat desenvolvimento/refatoracao/PROMPT_FASE_6_AUTOMATICO.md
# Enviar para Claude Code
# Configuração será automática
# Suporte contínuo é manual
```

---

## 📊 Resumo de Automação

| Fase | Automação | Requer Manual | Arquivo Prompt |
|------|-----------|---------------|----------------|
| **Fase 1** | ✅ 100% | Nenhum | ✅ Concluída |
| **Fase 2** | 🟢 100% | Nenhum | `PROMPT_FASE_2_AUTOMATICO.md` |
| **Fase 3** | 🟡 85% | Testes com falhas lógicas | `PROMPT_FASE_3_AUTOMATICO.md` |
| **Fase 4** | 🟢 100% | Nenhum | `PROMPT_FASE_4_AUTOMATICO.md` |
| **Fase 5** | 🔴 50% | Tokens PyPI, publicação | `PROMPT_FASE_5_AUTOMATICO.md` |
| **Fase 6** | 🟡 70% | Suporte contínuo | `PROMPT_FASE_6_AUTOMATICO.md` |

**Legenda:**
- 🟢 100% = Totalmente automático
- 🟡 70-85% = Maioria automática, alguma intervenção pode ser necessária
- 🔴 50% = Metade automática, metade manual

---

## 🚀 Como Usar Este Índice

### Para executar a próxima fase:

1. **Verificar pré-requisitos** da fase
2. **Ler o prompt** correspondente
3. **Copiar conteúdo** do arquivo PROMPT_FASE_X_AUTOMATICO.md
4. **Enviar para Claude Code**
5. **Aguardar execução** automática
6. **Verificar resultados** (checklist no final de cada prompt)

### Ordem de execução:

```
Fase 1 ✅ (Concluída)
  ↓
Fase 2 📦 (Próxima)
  ↓
Fase 3 🧪
  ↓
Fase 4 📝
  ↓
Fase 5 🚀 (requer tokens)
  ↓
Fase 6 🛠️ (configuração + suporte contínuo)
```

---

## 📁 Localização dos Arquivos

Todos os arquivos estão em:
```
/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/
```

**Arquivos originais (planejamento):**
- `FASE_1_PREPARACAO.md`
- `FASE_2_MIGRACAO_CODIGO.md`
- `FASE_3_MIGRACAO_TESTES.md`
- `FASE_4_DOCUMENTACAO.md`
- `FASE_5_RELEASE.md`
- `FASE_6_SUPORTE.md`

**Prompts executáveis (para IA):**
- `PROMPT_FASE_2_AUTOMATICO.md` ← **PRÓXIMO**
- `PROMPT_FASE_3_AUTOMATICO.md`
- `PROMPT_FASE_4_AUTOMATICO.md`
- `PROMPT_FASE_5_AUTOMATICO.md`
- `PROMPT_FASE_6_AUTOMATICO.md`

---

## ✅ Status Atual

- ✅ **Fase 1:** Concluída (2026-02-16)
- ⬜ **Fase 2:** Pronta para execução
- ⬜ **Fase 3:** Aguardando Fase 2
- ⬜ **Fase 4:** Aguardando Fase 3
- ⬜ **Fase 5:** Aguardando Fase 4
- ⬜ **Fase 6:** Aguardando Fase 5

---

## 🎯 Próxima Ação

**Execute a Fase 2:**

```bash
cat /home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/PROMPT_FASE_2_AUTOMATICO.md
```

Copie o conteúdo e envie para o Claude Code para execução 100% automática.

---

**Última atualização:** 2026-02-16
**Criado por:** Claude Code (Fase 1)
