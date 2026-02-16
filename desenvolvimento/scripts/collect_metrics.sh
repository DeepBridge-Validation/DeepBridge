#!/bin/bash

##############################################################################
# Script de Coleta de Métricas - DeepBridge
# 
# Coleta métricas de:
# - Downloads PyPI
# - Stars/Forks GitHub
# - Issues abertas/fechadas
# - Pull Requests
# - CI/CD Status
#
# Uso: ./scripts/collect_metrics.sh
##############################################################################

set -e

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   DeepBridge - Coleta de Métricas${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Verificar dependências
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI (gh) não encontrado. Instale com: sudo apt install gh${NC}"
    echo -e "${YELLOW}   Ou: https://cli.github.com/${NC}"
    GITHUB_AVAILABLE=false
else
    GITHUB_AVAILABLE=true
fi

if ! command -v curl &> /dev/null; then
    echo -e "${RED}❌ curl não encontrado. Instale com: sudo apt install curl${NC}"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠️  jq não encontrado (recomendado). Instale com: sudo apt install jq${NC}"
    JQ_AVAILABLE=false
else
    JQ_AVAILABLE=true
fi

echo ""

##############################################################################
# 1. MÉTRICAS DO PYPI
##############################################################################

echo -e "${GREEN}📦 Coletando métricas do PyPI...${NC}"
echo ""

# Função para obter stats do PyPI
get_pypi_stats() {
    local package=$1
    echo -e "${BLUE}  Package: ${package}${NC}"
    
    # Obter informações do pacote
    local response=$(curl -s "https://pypi.org/pypi/${package}/json")
    
    if [ $JQ_AVAILABLE = true ]; then
        local version=$(echo "$response" | jq -r '.info.version')
        local upload_time=$(echo "$response" | jq -r '.releases | to_entries | sort_by(.key) | last | .value[0].upload_time')
        
        echo "    Versão atual: $version"
        echo "    Último upload: $upload_time"
    else
        echo "    (jq não disponível - instale para ver mais detalhes)"
    fi
    
    # Downloads - usando pypistats (requer API)
    echo "    Downloads (últimos 30 dias): Consulte https://pypistats.org/packages/${package}"
    echo ""
}

# Coletar stats dos 3 pacotes
get_pypi_stats "deepbridge"
get_pypi_stats "deepbridge-distillation"
get_pypi_stats "deepbridge-synthetic"

##############################################################################
# 2. MÉTRICAS DO GITHUB
##############################################################################

if [ $GITHUB_AVAILABLE = true ]; then
    echo -e "${GREEN}⭐ Coletando métricas do GitHub...${NC}"
    echo ""
    
    # Função para obter stats do GitHub
    get_github_stats() {
        local repo=$1
        echo -e "${BLUE}  Repository: ${repo}${NC}"
        
        # Obter informações do repositório
        local repo_info=$(gh api repos/${repo} 2>/dev/null || echo "{}")
        
        if [ "$repo_info" != "{}" ] && [ $JQ_AVAILABLE = true ]; then
            local stars=$(echo "$repo_info" | jq -r '.stargazers_count')
            local forks=$(echo "$repo_info" | jq -r '.forks_count')
            local open_issues=$(echo "$repo_info" | jq -r '.open_issues_count')
            local watchers=$(echo "$repo_info" | jq -r '.watchers_count')
            
            echo "    ⭐ Stars: $stars"
            echo "    🍴 Forks: $forks"
            echo "    👁️  Watchers: $watchers"
            echo "    📋 Issues abertas: $open_issues"
            
            # Obter PRs abertas
            local open_prs=$(gh pr list --repo ${repo} --state open --json number --jq 'length' 2>/dev/null || echo "N/A")
            echo "    🔀 PRs abertas: $open_prs"
            
            # Obter status do CI (último workflow)
            local ci_status=$(gh run list --repo ${repo} --limit 1 --json conclusion --jq '.[0].conclusion' 2>/dev/null || echo "N/A")
            if [ "$ci_status" = "success" ]; then
                echo -e "    ✅ CI Status: ${GREEN}SUCCESS${NC}"
            elif [ "$ci_status" = "failure" ]; then
                echo -e "    ❌ CI Status: ${RED}FAILURE${NC}"
            else
                echo "    ⚠️  CI Status: $ci_status"
            fi
        else
            echo "    (Repositório não encontrado ou jq não disponível)"
        fi
        
        echo ""
    }
    
    # Coletar stats dos 3 repositórios
    get_github_stats "guhaase/deepbridge"
    get_github_stats "guhaase/deepbridge-distillation"
    get_github_stats "guhaase/deepbridge-synthetic"
else
    echo -e "${YELLOW}⚠️  GitHub CLI não disponível - pulando métricas do GitHub${NC}"
    echo ""
fi

##############################################################################
# 3. ISSUES E PRS (DETALHADO)
##############################################################################

if [ $GITHUB_AVAILABLE = true ]; then
    echo -e "${GREEN}📊 Análise de Issues e PRs...${NC}"
    echo ""
    
    analyze_issues() {
        local repo=$1
        echo -e "${BLUE}  Repository: ${repo}${NC}"
        
        # Issues abertas por label
        echo "    Issues abertas por tipo:"
        gh issue list --repo ${repo} --state open --json labels --jq '.[] | .labels[].name' 2>/dev/null | sort | uniq -c | while read count label; do
            echo "      - ${label}: ${count}"
        done
        
        # Issues fechadas recentemente (últimos 7 dias)
        local closed_count=$(gh issue list --repo ${repo} --state closed --search "closed:>=$(date -d '7 days ago' +%Y-%m-%d)" --json number --jq 'length' 2>/dev/null || echo "0")
        echo "    Issues fechadas (últimos 7 dias): $closed_count"
        
        echo ""
    }
    
    analyze_issues "guhaase/deepbridge"
    analyze_issues "guhaase/deepbridge-distillation"
    analyze_issues "guhaase/deepbridge-synthetic"
fi

##############################################################################
# 4. RESUMO
##############################################################################

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}✅ Coleta de métricas concluída!${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo "Para métricas de downloads detalhadas, visite:"
echo "  - https://pypistats.org/packages/deepbridge"
echo "  - https://pypistats.org/packages/deepbridge-distillation"
echo "  - https://pypistats.org/packages/deepbridge-synthetic"
echo ""
echo "Para análise completa do GitHub:"
echo "  - https://github.com/guhaase/deepbridge/pulse"
echo "  - https://github.com/guhaase/deepbridge/graphs/contributors"
echo ""

# Salvar métricas em arquivo com timestamp
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
METRICS_DIR="metrics"
mkdir -p "$METRICS_DIR"

echo "Métricas salvas em: ${METRICS_DIR}/metrics_${TIMESTAMP}.txt"
