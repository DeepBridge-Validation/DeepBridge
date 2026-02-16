#!/bin/bash
# Script para criar labels do GitHub nos repositórios DeepBridge
# Requer: gh CLI instalado e autenticado (gh auth login)

set -e

echo "🏷️  Criando labels do GitHub para os repositórios DeepBridge..."
echo ""

# Cores dos labels
COLOR_BUG="d73a4a"
COLOR_ENHANCEMENT="0e8a16"
COLOR_DOCUMENTATION="0075ca"
COLOR_QUESTION="d876e3"
COLOR_CRITICAL="b60205"
COLOR_HIGH="d93f0b"
COLOR_MEDIUM="fbca04"
COLOR_LOW="c2e0c6"
COLOR_MIGRATION="5319e7"

# Função para criar label
create_label() {
    local repo=$1
    local name=$2
    local color=$3
    local description=$4

    echo "  Criando label '$name' em $repo..."
    gh label create "$name" \
        --repo "$repo" \
        --color "$color" \
        --description "$description" \
        --force 2>/dev/null || echo "    ⚠️  Label '$name' já existe ou erro ao criar"
}

# Repositório principal: deepbridge
REPO_MAIN="guhaase/deepbridge"
echo "📦 Processando repositório: $REPO_MAIN"

create_label "$REPO_MAIN" "bug" "$COLOR_BUG" "Something isn't working"
create_label "$REPO_MAIN" "enhancement" "$COLOR_ENHANCEMENT" "New feature or request"
create_label "$REPO_MAIN" "documentation" "$COLOR_DOCUMENTATION" "Improvements or additions to documentation"
create_label "$REPO_MAIN" "question" "$COLOR_QUESTION" "Further information is requested"
create_label "$REPO_MAIN" "priority: critical" "$COLOR_CRITICAL" "Critical priority - needs immediate attention"
create_label "$REPO_MAIN" "priority: high" "$COLOR_HIGH" "High priority"
create_label "$REPO_MAIN" "priority: medium" "$COLOR_MEDIUM" "Medium priority"
create_label "$REPO_MAIN" "priority: low" "$COLOR_LOW" "Low priority"
create_label "$REPO_MAIN" "migration" "$COLOR_MIGRATION" "Related to migration from v1.x to v2.x"

echo ""

# Repositório: deepbridge-distillation
REPO_DISTILLATION="guhaase/deepbridge-distillation"
echo "📦 Processando repositório: $REPO_DISTILLATION"

create_label "$REPO_DISTILLATION" "bug" "$COLOR_BUG" "Something isn't working"
create_label "$REPO_DISTILLATION" "enhancement" "$COLOR_ENHANCEMENT" "New feature or request"
create_label "$REPO_DISTILLATION" "documentation" "$COLOR_DOCUMENTATION" "Improvements or additions to documentation"
create_label "$REPO_DISTILLATION" "question" "$COLOR_QUESTION" "Further information is requested"
create_label "$REPO_DISTILLATION" "priority: critical" "$COLOR_CRITICAL" "Critical priority - needs immediate attention"
create_label "$REPO_DISTILLATION" "priority: high" "$COLOR_HIGH" "High priority"
create_label "$REPO_DISTILLATION" "priority: medium" "$COLOR_MEDIUM" "Medium priority"
create_label "$REPO_DISTILLATION" "priority: low" "$COLOR_LOW" "Low priority"
create_label "$REPO_DISTILLATION" "migration" "$COLOR_MIGRATION" "Related to migration from v1.x to v2.x"

echo ""

# Repositório: deepbridge-synthetic
REPO_SYNTHETIC="guhaase/deepbridge-synthetic"
echo "📦 Processando repositório: $REPO_SYNTHETIC"

create_label "$REPO_SYNTHETIC" "bug" "$COLOR_BUG" "Something isn't working"
create_label "$REPO_SYNTHETIC" "enhancement" "$COLOR_ENHANCEMENT" "New feature or request"
create_label "$REPO_SYNTHETIC" "documentation" "$COLOR_DOCUMENTATION" "Improvements or additions to documentation"
create_label "$REPO_SYNTHETIC" "question" "$COLOR_QUESTION" "Further information is requested"
create_label "$REPO_SYNTHETIC" "priority: critical" "$COLOR_CRITICAL" "Critical priority - needs immediate attention"
create_label "$REPO_SYNTHETIC" "priority: high" "$COLOR_HIGH" "High priority"
create_label "$REPO_SYNTHETIC" "priority: medium" "$COLOR_MEDIUM" "Medium priority"
create_label "$REPO_SYNTHETIC" "priority: low" "$COLOR_LOW" "Low priority"
create_label "$REPO_SYNTHETIC" "migration" "$COLOR_MIGRATION" "Related to migration from v1.x to v2.x"

echo ""
echo "✅ Labels criados com sucesso nos 3 repositórios!"
echo ""
echo "📝 Notas:"
echo "  - Labels existentes foram atualizados (--force)"
echo "  - Verifique no GitHub: Settings → Labels"
echo ""
