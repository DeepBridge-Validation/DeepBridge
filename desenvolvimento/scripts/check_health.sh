#!/bin/bash

##############################################################################
# Script de Health Check - DeepBridge
# 
# Verifica:
# - Instalação dos pacotes
# - Dependências
# - Testes básicos de import
# - CI/CD status
# - Issues críticas
#
# Uso: ./scripts/check_health.sh
##############################################################################

set -e

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Contadores
PASS=0
FAIL=0
WARN=0

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   DeepBridge - Health Check${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

##############################################################################
# 1. VERIFICAR AMBIENTE PYTHON
##############################################################################

echo -e "${GREEN}🐍 Verificando ambiente Python...${NC}"
echo ""

# Python version
PYTHON_VERSION=$(python --version 2>&1)
echo -e "  Python: ${PYTHON_VERSION}"

if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "  ${GREEN}✅ Python 3.8+ detectado${NC}"
    ((PASS++))
else
    echo -e "  ${RED}❌ Python 3.8+ necessário${NC}"
    ((FAIL++))
fi

echo ""

##############################################################################
# 2. VERIFICAR INSTALAÇÃO DOS PACOTES
##############################################################################

echo -e "${GREEN}📦 Verificando instalação dos pacotes...${NC}"
echo ""

check_package() {
    local package=$1
    local import_name=$2
    
    if pip show "$package" &> /dev/null; then
        local version=$(pip show "$package" | grep Version | cut -d' ' -f2)
        echo -e "  ${GREEN}✅${NC} ${package}: ${version}"
        ((PASS++))
        
        # Tentar importar
        if python -c "import ${import_name}" 2>/dev/null; then
            echo -e "      Import OK: ${import_name}"
        else
            echo -e "      ${YELLOW}⚠️  Erro ao importar ${import_name}${NC}"
            ((WARN++))
        fi
    else
        echo -e "  ${RED}❌${NC} ${package}: NÃO INSTALADO"
        ((FAIL++))
    fi
}

check_package "deepbridge" "deepbridge"
check_package "deepbridge-distillation" "deepbridge.distillation"
check_package "deepbridge-synthetic" "deepbridge.synthetic"

echo ""

##############################################################################
# 3. VERIFICAR DEPENDÊNCIAS CRÍTICAS
##############################################################################

echo -e "${GREEN}🔧 Verificando dependências críticas...${NC}"
echo ""

check_dependency() {
    local package=$1
    local import_name=${2:-$1}
    local min_version=$3
    
    if python -c "import ${import_name}" 2>/dev/null; then
        local version=$(python -c "import ${import_name}; print(${import_name}.__version__)" 2>/dev/null || echo "unknown")
        echo -e "  ${GREEN}✅${NC} ${package}: ${version}"
        ((PASS++))
        
        # Verificar versão mínima se fornecida
        if [ ! -z "$min_version" ]; then
            if python -c "import ${import_name}; from packaging import version; import sys; sys.exit(0 if version.parse(${import_name}.__version__) >= version.parse('${min_version}') else 1)" 2>/dev/null; then
                echo -e "      (>= ${min_version} ✓)"
            else
                echo -e "      ${YELLOW}⚠️  Versão ${min_version}+ recomendada${NC}"
                ((WARN++))
            fi
        fi
    else
        echo -e "  ${RED}❌${NC} ${package}: NÃO INSTALADO"
        ((FAIL++))
    fi
}

check_dependency "torch" "torch" "2.0.0"
check_dependency "transformers" "transformers" "4.30.0"
check_dependency "numpy" "numpy"
check_dependency "tqdm" "tqdm"

echo ""

##############################################################################
# 4. TESTES BÁSICOS DE FUNCIONALIDADE
##############################################################################

echo -e "${GREEN}🧪 Executando testes básicos...${NC}"
echo ""

# Teste 1: Import básico
echo -e "  Teste 1: Import básico do DeepBridge"
if python -c "
import deepbridge
from deepbridge.distillation import DistillationTrainer
from deepbridge.synthetic import SyntheticDataGenerator
print('  Imports OK')
" 2>/dev/null; then
    echo -e "  ${GREEN}✅ Imports funcionando${NC}"
    ((PASS++))
else
    echo -e "  ${RED}❌ Erro nos imports${NC}"
    ((FAIL++))
fi

# Teste 2: Verificar versões consistentes
echo -e "  Teste 2: Consistência de versões"
if python -c "
import deepbridge
import deepbridge.distillation
import deepbridge.synthetic

base_version = deepbridge.__version__
distill_version = deepbridge.distillation.__version__
synthetic_version = deepbridge.synthetic.__version__

if base_version == distill_version == synthetic_version:
    print(f'  Versões consistentes: {base_version}')
else:
    print(f'  AVISO: Versões diferentes!')
    print(f'    deepbridge: {base_version}')
    print(f'    distillation: {distill_version}')
    print(f'    synthetic: {synthetic_version}')
    exit(1)
" 2>/dev/null; then
    echo -e "  ${GREEN}✅ Versões consistentes${NC}"
    ((PASS++))
else
    echo -e "  ${YELLOW}⚠️  Versões inconsistentes entre pacotes${NC}"
    ((WARN++))
fi

echo ""

##############################################################################
# 5. VERIFICAR GITHUB STATUS (SE DISPONÍVEL)
##############################################################################

if command -v gh &> /dev/null; then
    echo -e "${GREEN}🔍 Verificando status do GitHub...${NC}"
    echo ""
    
    check_repo_health() {
        local repo=$1
        echo -e "  ${BLUE}Repository: ${repo}${NC}"
        
        # CI Status
        local ci_status=$(gh run list --repo ${repo} --limit 1 --json conclusion --jq '.[0].conclusion' 2>/dev/null || echo "unknown")
        if [ "$ci_status" = "success" ]; then
            echo -e "    ${GREEN}✅ CI: SUCCESS${NC}"
            ((PASS++))
        elif [ "$ci_status" = "failure" ]; then
            echo -e "    ${RED}❌ CI: FAILURE${NC}"
            ((FAIL++))
        else
            echo -e "    ${YELLOW}⚠️  CI: ${ci_status}${NC}"
            ((WARN++))
        fi
        
        # Issues críticas
        local critical_issues=$(gh issue list --repo ${repo} --label "priority:critical" --state open --json number --jq 'length' 2>/dev/null || echo "0")
        if [ "$critical_issues" -eq 0 ]; then
            echo -e "    ${GREEN}✅ Nenhuma issue crítica aberta${NC}"
            ((PASS++))
        else
            echo -e "    ${RED}❌ ${critical_issues} issue(s) crítica(s) aberta(s)${NC}"
            ((FAIL++))
        fi
        
        # PRs pendentes
        local open_prs=$(gh pr list --repo ${repo} --state open --json number --jq 'length' 2>/dev/null || echo "0")
        echo -e "    📋 PRs abertas: ${open_prs}"
        
        echo ""
    }
    
    check_repo_health "guhaase/deepbridge"
    check_repo_health "guhaase/deepbridge-distillation"
    check_repo_health "guhaase/deepbridge-synthetic"
else
    echo -e "${YELLOW}⚠️  GitHub CLI não disponível - pulando verificações do GitHub${NC}"
    echo ""
fi

##############################################################################
# 6. VERIFICAR ARQUIVOS DE CONFIGURAÇÃO
##############################################################################

echo -e "${GREEN}📁 Verificando arquivos de configuração...${NC}"
echo ""

check_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅${NC} ${file}"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠️${NC}  ${file} (não encontrado)"
        ((WARN++))
    fi
}

check_file "setup.py"
check_file "pyproject.toml"
check_file "README.md"
check_file ".github/workflows/ci.yml"

echo ""

##############################################################################
# 7. RESUMO
##############################################################################

TOTAL=$((PASS + FAIL + WARN))

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}📊 Resumo do Health Check${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo -e "  ${GREEN}✅ Testes passados: ${PASS}${NC}"
echo -e "  ${YELLOW}⚠️  Avisos: ${WARN}${NC}"
echo -e "  ${RED}❌ Falhas: ${FAIL}${NC}"
echo -e "  📊 Total de verificações: ${TOTAL}"
echo ""

# Determinar status geral
if [ $FAIL -eq 0 ] && [ $WARN -eq 0 ]; then
    echo -e "${GREEN}🎉 Sistema 100% saudável!${NC}"
    exit 0
elif [ $FAIL -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Sistema OK com avisos${NC}"
    exit 0
else
    echo -e "${RED}❌ Sistema com problemas - ação necessária${NC}"
    exit 1
fi
