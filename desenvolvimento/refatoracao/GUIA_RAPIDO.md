# 🚀 Guia Rápido - Publicar DeepBridge v2.0.0

## ⚡ INÍCIO RÁPIDO (5 minutos de leitura)

```
┌──────────────────────────────────────────────────────────────┐
│  ✅ 71% Concluído Automaticamente (58/82 checkboxes)       │
│  ⚠️  29% Aguardando Tokens (24/82 checkboxes)              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📝 O QUE VOCÊ PRECISA FAZER

### 1️⃣ Obter Tokens (5-10 minutos)

#### Test PyPI (Recomendado - teste antes de publicar de verdade)
```
🔗 https://test.pypi.org/account/register/
   ↓
🔗 https://test.pypi.org/manage/account/token/
   ↓
📋 Copiar token (começa com: pypi-...)
```

#### PyPI Oficial (Produção - IRREVERSÍVEL)
```
🔗 https://pypi.org/account/register/
   ↓
🔗 https://pypi.org/manage/account/token/
   ↓
📋 Copiar token (começa com: pypi-...)
```

### 2️⃣ Executar Script (1 comando!)

```bash
cd /home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao
./publicar_pypi.sh
```

**O script faz TUDO automaticamente**:
- ✓ Solicita os tokens
- ✓ Configura o Poetry
- ✓ Publica os 3 pacotes
- ✓ Testa as instalações
- ✓ Marca os checkboxes no documento
- ✓ Mostra URLs para verificação

---

## 🎯 FLUXO RECOMENDADO

```
┌─────────────────────────────────────────────────────────────┐
│  ETAPA 1: Test PyPI (Ambiente de Testes)                   │
│  ↓                                                           │
│  • Execute: ./publicar_pypi.sh → Opção 1                   │
│  • Cole seu token do Test PyPI quando solicitado           │
│  • Aguarde publicação dos 3 pacotes                        │
│  • Teste automático da instalação                          │
│  • Verifique URLs no Test PyPI                             │
│                                                              │
│  ✅ Se tudo OK, continue para Etapa 2                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ETAPA 2: PyPI Oficial (Produção - IRREVERSÍVEL!)          │
│  ↓                                                           │
│  • Execute: ./publicar_pypi.sh → Opção 2                   │
│  • Confirme que deseja publicar (digite: sim)              │
│  • Cole seu token do PyPI oficial quando solicitado        │
│  • Aguarde publicação dos 3 pacotes                        │
│  • Teste automático da instalação                          │
│  • Verifique URLs no PyPI oficial                          │
│                                                              │
│  ✅ Pacotes disponíveis publicamente!                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ETAPA 3: Deprecação v1.x                                   │
│  ↓                                                           │
│  • Execute: ./publicar_pypi.sh → Opção 3                   │
│  • Publica v1.63.0 com deprecation warning                 │
│                                                              │
│  ✅ Usuários v1.x verão aviso de migração                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ETAPA 4: GitHub Releases                                   │
│  ↓                                                           │
│  • Execute: gh auth login (se não autenticado)             │
│  • Execute: ./publicar_pypi.sh → Opção 4                   │
│  • Releases criados automaticamente para os 3 repos       │
│                                                              │
│  ✅ Releases visíveis no GitHub!                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⏱️ TEMPO TOTAL: ~25 minutos

- Obter tokens: ~5 min
- Test PyPI: ~5 min
- PyPI oficial: ~5 min
- v1.63.0: ~2 min
- GitHub Releases: ~3 min
- Verificações: ~5 min

---

## 🔍 VERIFICAR APÓS PUBLICAÇÃO

### URLs para Verificar:

**PyPI Oficial**:
- https://pypi.org/project/deepbridge/
- https://pypi.org/project/deepbridge-distillation/
- https://pypi.org/project/deepbridge-synthetic/

**GitHub Releases**:
- https://github.com/guhaase/DeepBridge/releases/tag/v2.0.0
- https://github.com/[seu-usuario]/deepbridge-distillation/releases/tag/v2.0.0
- https://github.com/[seu-usuario]/deepbridge-synthetic/releases/tag/v2.0.0

### Testar Instalação:

```bash
# Criar ambiente limpo
python -m venv /tmp/test_final
source /tmp/test_final/bin/activate

# Instalar pacotes
pip install deepbridge deepbridge-distillation deepbridge-synthetic

# Testar imports
python -c "import deepbridge, deepbridge_distillation, deepbridge_synthetic; print('✓ Tudo OK!')"

# Limpar
deactivate
rm -rf /tmp/test_final
```

---

## 📊 ARQUIVOS CRIADOS PARA VOCÊ

1. **`publicar_pypi.sh`** ⭐ (NOVO!)
   - Script interativo principal
   - Marca checkboxes automaticamente
   - **USE ESTE!**

2. **`RELATORIO_PENDENCIAS.md`**
   - Detalhamento completo das tarefas
   - Status de cada checkbox
   - Instruções detalhadas

3. **`GUIA_RAPIDO.md`** (este arquivo)
   - Resumo visual rápido
   - Fluxo de trabalho
   - Comandos prontos

4. **`INSTRUCOES_PUBLICACAO_MANUAL.md`**
   - Alternativa manual
   - Comandos individuais
   - Troubleshooting

---

## ❓ PERGUNTAS FREQUENTES

### "Posso pular o Test PyPI?"

Não recomendado. O Test PyPI permite:
- Testar a publicação sem comprometer o PyPI oficial
- Verificar se os pacotes instalam corretamente
- Validar dependências
- Corrigir erros antes de publicar oficialmente

### "O que acontece se eu errar no PyPI oficial?"

**Você NÃO pode**:
- Excluir uma versão publicada
- Sobrescrever uma versão existente
- Fazer upload novamente do mesmo arquivo

**Você pode apenas**:
- Publicar uma nova versão corrigida (ex: 2.0.1)
- Ocultar a versão problemática (yank)

Por isso: **SEMPRE teste no Test PyPI primeiro!**

### "Preciso fazer tudo de uma vez?"

Não! O script oferece opções separadas:
1. Apenas Test PyPI
2. Apenas PyPI oficial
3. Apenas v1.63.0
4. Apenas GitHub Releases
5. Tudo em sequência (recomendado)

### "Os checkboxes serão marcados automaticamente?"

**SIM!** O script `publicar_pypi.sh` marca automaticamente os checkboxes no arquivo `PROMPT_FASE_5_AUTOMATICO.md` conforme você completa cada tarefa.

---

## 🆘 PROBLEMAS COMUNS

### Token inválido
```
Erro: HTTP 403 - Invalid authentication

Solução:
1. Verifique se copiou o token completo
2. Token deve começar com: pypi-
3. Gere um novo token se necessário
```

### Pacote já existe
```
Erro: File already exists

Solução:
1. Versão já foi publicada
2. Não é possível sobrescrever
3. Incremente a versão (ex: 2.0.1)
```

### GitHub CLI não autenticado
```
Erro: authentication required

Solução:
1. Execute: gh auth login
2. Siga as instruções no terminal
3. Execute o script novamente
```

---

## ✅ CHECKLIST PESSOAL

Marque à medida que completa:

```
□ Ler este guia rápido
□ Obter token do Test PyPI
□ Executar: ./publicar_pypi.sh → Opção 1
□ Verificar pacotes no Test PyPI
□ Testar instalação do Test PyPI
□ Obter token do PyPI oficial
□ Executar: ./publicar_pypi.sh → Opção 2
□ Verificar pacotes no PyPI oficial
□ Executar: ./publicar_pypi.sh → Opção 3 (v1.63.0)
□ Executar: gh auth login
□ Executar: ./publicar_pypi.sh → Opção 4 (Releases)
□ Testar instalação final
□ Verificar todos os checkboxes marcados no PROMPT_FASE_5_AUTOMATICO.md
□ Comemorar! 🎉
```

---

## 🎉 SUCESSO!

Quando concluir, você terá:

✓ DeepBridge v2.0.0 publicado no PyPI
✓ deepbridge-distillation v2.0.0 publicado no PyPI
✓ deepbridge-synthetic v2.0.0 publicado no PyPI
✓ v1.63.0 com deprecation warning
✓ Releases criados no GitHub
✓ 82/82 checkboxes marcados (100%)
✓ Fase 5 CONCLUÍDA!

---

**COMECE AGORA**: `./publicar_pypi.sh`

Tempo estimado: 25 minutos até conclusão completa 🚀
