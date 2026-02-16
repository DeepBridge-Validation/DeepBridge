#!/bin/bash
# Script para criar labels padronizados nos repositórios DeepBridge
# Requer: gh CLI instalado e autenticado (gh auth login)

set -e

echo "=========================================="
echo "  Criador de Labels GitHub - DeepBridge"
echo "=========================================="
echo ""

# Cores dos labels (formato hex sem #)
COLOR_BUG="d73a4a"
COLOR_ENHANCEMENT="0e8a16"
COLOR_DOCUMENTATION="0075ca"
COLOR_QUESTION="d876e3"
COLOR_MIGRATION="5319e7"
COLOR_CRITICAL="b60205"
COLOR_HIGH="d93f0b"
COLOR_MEDIUM="fbca04"
COLOR_LOW="c2e0c6"

# Repositórios
REPOS=(
    "DeepBridge-Validation/DeepBridge"
    "DeepBridge-Validation/deepbridge-distillation"
    "DeepBridge-Validation/deepbridge-synthetic"
)

# Verificar se gh está instalado
if ! command -v gh &> /dev/null; then
    echo "❌ Erro: GitHub CLI (gh) não está instalado"
    echo ""
    echo "Instale com:"
    echo "  Ubuntu/Debian: sudo apt install gh"
    echo "  macOS: brew install gh"
    echo "  Windows: winget install GitHub.cli"
    echo ""
    exit 1
fi

# Verificar se está autenticado
if ! gh auth status &> /dev/null; then
    echo "❌ Erro: Não autenticado no GitHub CLI"
    echo ""
    echo "Execute: gh auth login"
    echo ""
    exit 1
fi

echo "✅ GitHub CLI instalado e autenticado"
echo ""

# Função para criar label
create_label() {
    local repo=$1
    local name=$2
    local color=$3
    local description=$4

    echo "  Criando label: $name"

    # Usar --force para atualizar se já existir
    if gh label create "$name" \
        --repo "$repo" \
        --color "$color" \
        --description "$description" \
        --force 2>/dev/null; then
        echo "    ✅ Criado/Atualizado"
    else
        echo "    ⚠️  Aviso: Não foi possível criar (pode já existir)"
    fi
}

# Criar labels em cada repositório
for repo in "${REPOS[@]}"; do
    echo "=========================================="
    echo "Repositório: $repo"
    echo "=========================================="
    echo ""

    # Labels de tipo
    create_label "$repo" "bug" "$COLOR_BUG" "Something isn't working"
    create_label "$repo" "enhancement" "$COLOR_ENHANCEMENT" "New feature or request"
    create_label "$repo" "documentation" "$COLOR_DOCUMENTATION" "Improvements or additions to documentation"
    create_label "$repo" "question" "$COLOR_QUESTION" "Further information is requested"
    create_label "$repo" "migration" "$COLOR_MIGRATION" "Related to migration from v1.x to v2.x"

    # Labels de prioridade
    create_label "$repo" "priority: critical" "$COLOR_CRITICAL" "Critical priority - needs immediate attention"
    create_label "$repo" "priority: high" "$COLOR_HIGH" "High priority"
    create_label "$repo" "priority: medium" "$COLOR_MEDIUM" "Medium priority"
    create_label "$repo" "priority: low" "$COLOR_LOW" "Low priority"

    echo ""
done

echo "=========================================="
echo "✅ Labels criados com sucesso!"
echo "=========================================="
echo ""
echo "Próximos passos:"
echo "1. Verifique os labels em cada repositório:"
echo "   - https://github.com/DeepBridge-Validation/DeepBridge/labels"
echo "   - https://github.com/DeepBridge-Validation/deepbridge-distillation/labels"
echo "   - https://github.com/DeepBridge-Validation/deepbridge-synthetic/labels"
echo ""
echo "2. Os labels estarão disponíveis ao criar issues"
echo ""
