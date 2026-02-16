#!/bin/bash

################################################################################
# DeepBridge - Health Check Script
#
# This script performs health checks on DeepBridge packages:
# - PyPI availability
# - GitHub repository status
# - CI/CD pipeline status
# - Documentation availability
# - Issue tracker health
#
# Usage:
#   ./scripts/check_health.sh [--detailed] [--json]
#
# Exit codes:
#   0 - All checks passed
#   1 - Some checks failed
#   2 - Critical failures detected
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DETAILED="${DETAILED:-0}"
JSON_OUTPUT="${JSON_OUTPUT:-0}"

# Package names
PACKAGES=(
    "deepbridge"
    "deepbridge-distillation"
    "deepbridge-synthetic"
)

# GitHub repositories
GITHUB_REPOS=(
    "guhaase/DeepBridge"
    "guhaase/deepbridge-distillation"
    "guhaase/deepbridge-synthetic"
)

# Documentation URLs
DOC_URLS=(
    "https://deepbridge.readthedocs.io/"
    "https://deepbridge-distillation.readthedocs.io/"
    "https://deepbridge-synthetic.readthedocs.io/"
)

# Health tracking
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
CRITICAL_FAILURES=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log_info() {
    if [ "$JSON_OUTPUT" -eq 0 ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    if [ "$JSON_OUTPUT" -eq 0 ]; then
        echo -e "${GREEN}[✓]${NC} $1"
    fi
}

log_warning() {
    if [ "$JSON_OUTPUT" -eq 0 ]; then
        echo -e "${YELLOW}[⚠]${NC} $1"
    fi
}

log_error() {
    if [ "$JSON_OUTPUT" -eq 0 ]; then
        echo -e "${RED}[✗]${NC} $1"
    fi
}

log_critical() {
    if [ "$JSON_OUTPUT" -eq 0 ]; then
        echo -e "${RED}[CRITICAL]${NC} $1"
    fi
    ((CRITICAL_FAILURES++))
}

record_check() {
    local result=$1
    ((TOTAL_CHECKS++))
    if [ "$result" -eq 0 ]; then
        ((PASSED_CHECKS++))
    else
        ((FAILED_CHECKS++))
    fi
}

################################################################################
# Health Check Functions
################################################################################

check_pypi_availability() {
    local package=$1
    log_info "Checking PyPI availability for $package..."

    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/$package/json")

    if [ "$status_code" -eq 200 ]; then
        log_success "PyPI: $package is available (HTTP $status_code)"
        record_check 0
        return 0
    else
        log_error "PyPI: $package is NOT available (HTTP $status_code)"
        record_check 1
        return 1
    fi
}

check_pypi_version() {
    local package=$1
    log_info "Checking latest version of $package on PyPI..."

    local response
    response=$(curl -s "https://pypi.org/pypi/$package/json" || echo "{}")

    if command -v jq &> /dev/null; then
        local version=$(echo "$response" | jq -r '.info.version // "unknown"')

        if [ "$version" != "unknown" ] && [ "$version" != "null" ]; then
            log_success "Latest version: $version"
            record_check 0
            return 0
        else
            log_error "Could not determine version"
            record_check 1
            return 1
        fi
    else
        log_warning "jq not installed, skipping version check"
        record_check 0
        return 0
    fi
}

check_github_availability() {
    local repo=$1
    log_info "Checking GitHub repository: $repo..."

    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/$repo")

    if [ "$status_code" -eq 200 ]; then
        log_success "GitHub: $repo is accessible (HTTP $status_code)"
        record_check 0
        return 0
    else
        log_critical "GitHub: $repo is NOT accessible (HTTP $status_code)"
        record_check 1
        return 1
    fi
}

check_github_issues() {
    local repo=$1

    if ! command -v jq &> /dev/null; then
        log_warning "jq not installed, skipping issue check"
        return 0
    fi

    log_info "Checking open issues for $repo..."

    local response
    response=$(curl -s "https://api.github.com/repos/$repo" || echo "{}")

    local open_issues=$(echo "$response" | jq -r '.open_issues_count // -1')

    if [ "$open_issues" -ge 0 ]; then
        if [ "$open_issues" -gt 20 ]; then
            log_warning "High number of open issues: $open_issues"
        else
            log_success "Open issues: $open_issues"
        fi
        record_check 0
        return 0
    else
        log_error "Could not fetch issue count"
        record_check 1
        return 1
    fi
}

check_ci_status() {
    local repo=$1

    if ! command -v gh &> /dev/null; then
        log_warning "gh CLI not installed, skipping CI check"
        return 0
    fi

    log_info "Checking CI/CD status for $repo..."

    local latest_run
    latest_run=$(gh run list --repo "$repo" --limit 1 --json status,conclusion 2>/dev/null || echo "[]")

    if [ "$latest_run" = "[]" ]; then
        log_warning "No CI runs found or unable to fetch"
        record_check 0
        return 0
    fi

    if command -v jq &> /dev/null; then
        local status=$(echo "$latest_run" | jq -r '.[0].status // "unknown"')
        local conclusion=$(echo "$latest_run" | jq -r '.[0].conclusion // "unknown"')

        if [ "$conclusion" = "success" ]; then
            log_success "Latest CI run: $conclusion"
            record_check 0
            return 0
        elif [ "$conclusion" = "failure" ]; then
            log_error "Latest CI run: $conclusion"
            record_check 1
            return 1
        else
            log_warning "Latest CI run: $status ($conclusion)"
            record_check 0
            return 0
        fi
    fi
}

check_documentation() {
    local url=$1
    local package=$2

    log_info "Checking documentation for $package..."

    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

    if [ "$status_code" -eq 200 ]; then
        log_success "Documentation accessible: $url"
        record_check 0
        return 0
    else
        log_warning "Documentation may not be available: $url (HTTP $status_code)"
        record_check 1
        return 1
    fi
}

################################################################################
# Detailed Health Checks
################################################################################

run_detailed_checks() {
    log_info "Running detailed health checks..."
    echo ""

    # Check if installed locally
    for package in "${PACKAGES[@]}"; do
        log_info "Checking local installation of $package..."

        if python3 -c "import ${package//-/_}" 2>/dev/null; then
            log_success "$package is installed locally"

            local version=$(python3 -c "import ${package//-/_}; print(${package//-/_}.__version__)" 2>/dev/null || echo "unknown")
            log_info "  Version: $version"
        else
            log_warning "$package is NOT installed locally"
        fi
        echo ""
    done

    # Check import health
    log_info "Testing imports..."

    if python3 -c "from deepbridge import Bridge" 2>/dev/null; then
        log_success "Core imports working"
    else
        log_warning "Core imports may have issues"
    fi

    if python3 -c "from deepbridge.distillation import KnowledgeDistiller" 2>/dev/null; then
        log_success "Distillation imports working"
    else
        log_warning "Distillation module not installed or has issues"
    fi

    if python3 -c "from deepbridge.synthetic import SyntheticDataGenerator" 2>/dev/null; then
        log_success "Synthetic imports working"
    else
        log_warning "Synthetic module not installed or has issues"
    fi

    echo ""
}

################################################################################
# Main Health Check Logic
################################################################################

run_health_checks() {
    local start_time=$(date +%s)

    if [ "$JSON_OUTPUT" -eq 0 ]; then
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║        DeepBridge Ecosystem Health Check                      ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
    fi

    # PyPI Health Checks
    for package in "${PACKAGES[@]}"; do
        check_pypi_availability "$package"
        check_pypi_version "$package"
        echo ""
    done

    # GitHub Health Checks
    for repo in "${GITHUB_REPOS[@]}"; do
        check_github_availability "$repo"
        check_github_issues "$repo"
        check_ci_status "$repo"
        echo ""
    done

    # Documentation Health Checks
    for i in "${!PACKAGES[@]}"; do
        check_documentation "${DOC_URLS[$i]}" "${PACKAGES[$i]}"
    done

    echo ""

    # Detailed checks if requested
    if [ "$DETAILED" -eq 1 ]; then
        run_detailed_checks
    fi

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ "$JSON_OUTPUT" -eq 1 ]; then
        cat <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_seconds": $duration,
  "total_checks": $TOTAL_CHECKS,
  "passed": $PASSED_CHECKS,
  "failed": $FAILED_CHECKS,
  "critical_failures": $CRITICAL_FAILURES,
  "health_status": "$([ $CRITICAL_FAILURES -eq 0 ] && [ $FAILED_CHECKS -eq 0 ] && echo "healthy" || [ $CRITICAL_FAILURES -eq 0 ] && echo "degraded" || echo "critical")"
}
EOF
    else
        echo "════════════════════════════════════════════════════════════════"
        echo "Health Check Summary:"
        echo "────────────────────────────────────────────────────────────────"
        echo "Total Checks:        $TOTAL_CHECKS"
        echo -e "Passed:              ${GREEN}$PASSED_CHECKS${NC}"
        echo -e "Failed:              ${YELLOW}$FAILED_CHECKS${NC}"
        echo -e "Critical Failures:   ${RED}$CRITICAL_FAILURES${NC}"
        echo "Duration:            ${duration}s"
        echo "════════════════════════════════════════════════════════════════"

        if [ $CRITICAL_FAILURES -gt 0 ]; then
            echo -e "${RED}STATUS: CRITICAL - Immediate attention required${NC}"
        elif [ $FAILED_CHECKS -gt 0 ]; then
            echo -e "${YELLOW}STATUS: DEGRADED - Some issues detected${NC}"
        else
            echo -e "${GREEN}STATUS: HEALTHY - All checks passed${NC}"
        fi
        echo ""
    fi

    # Exit code
    if [ $CRITICAL_FAILURES -gt 0 ]; then
        return 2
    elif [ $FAILED_CHECKS -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --detailed|-d)
                DETAILED=1
                shift
                ;;
            --json)
                JSON_OUTPUT=1
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--detailed] [--json]"
                echo ""
                echo "Options:"
                echo "  --detailed, -d   Run detailed health checks"
                echo "  --json           Output results in JSON format"
                echo "  --help, -h       Show this help message"
                echo ""
                echo "Exit codes:"
                echo "  0 - All checks passed"
                echo "  1 - Some checks failed"
                echo "  2 - Critical failures detected"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run health checks
    run_health_checks
    exit $?
}

# Run main function
main "$@"
