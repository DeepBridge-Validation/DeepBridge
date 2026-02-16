# Estrutura de Repositórios - DeepBridge v2.0

**Data:** 2026-02-16
**Versão:** 1.0

---

## 📁 Estrutura Recomendada

Todos os 3 repositórios ficam organizados dentro de uma pasta `deepbridge_toolkit`:

```
/home/guhaase/projetos/deepbridge_toolkit/
├── DeepBridge/                    # Repo principal (core - validation)
│   ├── deepbridge/
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── deepbridge-distillation/       # Extensão - Model distillation
│   ├── deepbridge_distillation/
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
└── deepbridge-synthetic/          # Standalone - Synthetic data
    ├── deepbridge_synthetic/
    ├── tests/
    ├── pyproject.toml
    └── README.md
```

---

## 🎯 Vantagens desta Estrutura

1. **Organização Clara**
   - Todos os repos relacionados ao DeepBridge em um só lugar
   - Fácil navegação entre projetos
   - Estrutura hierárquica lógica

2. **Facilita Desenvolvimento**
   - Trabalhar em múltiplos repos simultaneamente
   - Testar integração entre pacotes
   - Gerenciar dependências locais

3. **Simplicidade**
   - Um único diretório toolkit para todos os projetos
   - Fácil de fazer backup
   - Fácil de compartilhar com equipe

4. **Escalabilidade**
   - Adicionar novos pacotes no futuro (ex: deepbridge-explainability)
   - Manter estrutura consistente
   - Facilita CI/CD e automação

---

## 🔄 Migração da Estrutura Atual

Se o repo `DeepBridge` já existe em `/home/guhaase/projetos/DeepBridge`, você tem 2 opções:

### Opção 1: Mover para dentro do toolkit (RECOMENDADO)

```bash
# Criar diretório toolkit
mkdir -p /home/guhaase/projetos/deepbridge_toolkit

# Mover DeepBridge para dentro
mv /home/guhaase/projetos/DeepBridge /home/guhaase/projetos/deepbridge_toolkit/

# Resultado:
# /home/guhaase/projetos/deepbridge_toolkit/DeepBridge/
```

**Vantagens:**
- ✅ Tudo no mesmo lugar
- ✅ Mais organizado
- ✅ Facilita trabalho simultâneo

**Desvantagens:**
- ⚠️ Precisa atualizar paths no IDE/editor
- ⚠️ Histórico de comandos pode ter path antigo

### Opção 2: Deixar DeepBridge fora, novos repos dentro

```bash
# Criar toolkit e clonar apenas os novos
mkdir -p /home/guhaase/projetos/deepbridge_toolkit
cd /home/guhaase/projetos/deepbridge_toolkit

git clone https://github.com/DeepBridge-Validation/deepbridge-distillation.git
git clone https://github.com/DeepBridge-Validation/deepbridge-synthetic.git

# Resultado:
# /home/guhaase/projetos/DeepBridge/                     (core - fora)
# /home/guhaase/projetos/deepbridge_toolkit/
#   ├── deepbridge-distillation/
#   └── deepbridge-synthetic/
```

**Vantagens:**
- ✅ Não precisa mover nada
- ✅ Paths do DeepBridge não mudam

**Desvantagens:**
- ⚠️ Menos organizado
- ⚠️ Precisa navegar para 2 locais diferentes

---

## 📝 Atualização de Documentação

Todos os prompts de execução automática foram atualizados para usar:

```
/home/guhaase/projetos/deepbridge_toolkit/
```

Arquivos atualizados:
- ✅ `PROMPT_FASE_2_AUTOMATICO.md`
- ✅ `FASE_2_MIGRACAO_CODIGO.md`

---

## 🚀 Próximos Passos

1. **Escolher opção** (mover DeepBridge ou não)
2. **Executar Fase 2** com a nova estrutura
3. **Validar** que tudo funciona

---

## 💡 Recomendação

**Use a Opção 1** (mover DeepBridge para dentro do toolkit).

É mais organizado e facilita o desenvolvimento a longo prazo. Atualizar o path no IDE é rápido e vale a pena pela organização.

---

**Criado por:** Claude Code
**Última atualização:** 2026-02-16
