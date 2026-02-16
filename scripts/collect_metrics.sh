#!/bin/bash

################################################################################
# DeepBridge - Metrics Collection Script
#
# This script collects various metrics from PyPI, GitHub, and CI/CD
# to monitor the health and adoption of DeepBridge packages.
#
# Usage:
#   ./scripts/collect_metrics.sh [--output metrics.json] [--verbose]
#
# Requirements:
#   - curl (for API calls)
#   - jq (for JSON parsing) - optional but recommended
#   - gh CLI (for GitHub metrics) - optional
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="${OUTPUT_FILE:-metrics_$(date +%Y%m%d_%H%M%S).json}"
VERBOSE="${VERBOSE:-0}"

# Package names
PACKAGES=(
    "deepbridge"
    "deepbridge-distillation"
    "deepbridge-synthetic"
)

# GitHub repositories (format: owner/repo)
GITHUB_REPOS=(
    "guhaase/DeepBridge"
    "guhaase/deepbridge-distillation"
    "guhaase/deepbridge-synthetic"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" -eq 1 ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

check_dependencies() {
    local missing_deps=0

    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed"
        missing_deps=1
    fi

    if ! command -v jq &> /dev/null; then
        log_warning "jq is not installed - JSON output will be less formatted"
    fi

    if ! command -v gh &> /dev/null; then
        log_warning "gh CLI is not installed - GitHub metrics will be limited"
    fi

    return $missing_deps
}

################################################################################
# PyPI Metrics Collection
################################################################################

get_pypi_metrics() {
    local package=$1
    log_info "Collecting PyPI metrics for $package..."

    # Fetch package info from PyPI JSON API
    local response
    response=$(curl -s "https://pypi.org/pypi/$package/json" || echo "{}")

    if [ "$response" = "{}" ]; then
        log_warning "Failed to fetch PyPI data for $package"
        echo "{}"
        return 1
    fi

    # Extract metrics using jq if available, otherwise use grep/sed
    if command -v jq &> /dev/null; then
        local version=$(echo "$response" | jq -r '.info.version // "unknown"')
        local summary=$(echo "$response" | jq -r '.info.summary // "N/A"')
        local author=$(echo "$response" | jq -r '.info.author // "N/A"')
        local license=$(echo "$response" | jq -r '.info.license // "N/A"')
        local python_requires=$(echo "$response" | jq -r '.info.requires_python // "N/A"')
        local upload_time=$(echo "$response" | jq -r '.urls[0].upload_time // "N/A"')

        log_verbose "  Version: $version"
        log_verbose "  Latest upload: $upload_time"

        cat <<EOF
{
  "package": "$package",
  "version": "$version",
  "summary": "$summary",
  "author": "$author",
  "license": "$license",
  "python_requires": "$python_requires",
  "upload_time": "$upload_time",
  "pypi_url": "https://pypi.org/project/$package/"
}
EOF
    else
        # Fallback without jq
        cat <<EOF
{
  "package": "$package",
  "pypi_url": "https://pypi.org/project/$package/",
  "note": "Install jq for detailed metrics"
}
EOF
    fi
}

################################################################################
# GitHub Metrics Collection
################################################################################

get_github_metrics() {
    local repo=$1
    log_info "Collecting GitHub metrics for $repo..."

    # Try using gh CLI first
    if command -v gh &> /dev/null; then
        local repo_info
        repo_info=$(gh repo view "$repo" --json stargazerCount,forkCount,openIssues,watchers,createdAt,updatedAt,description 2>/dev/null || echo "{}")

        if [ "$repo_info" != "{}" ]; then
            log_verbose "  Using gh CLI for GitHub metrics"
            echo "$repo_info"
            return 0
        fi
    fi

    # Fallback to GitHub API
    local response
    response=$(curl -s "https://api.github.com/repos/$repo" || echo "{}")

    if [ "$response" = "{}" ]; then
        log_warning "Failed to fetch GitHub data for $repo"
        echo "{}"
        return 1
    fi

    if command -v jq &> /dev/null; then
        local stars=$(echo "$response" | jq -r '.stargazers_count // 0')
        local forks=$(echo "$response" | jq -r '.forks_count // 0')
        local open_issues=$(echo "$response" | jq -r '.open_issues_count // 0')
        local watchers=$(echo "$response" | jq -r '.watchers_count // 0')
        local created_at=$(echo "$response" | jq -r '.created_at // "N/A"')
        local updated_at=$(echo "$response" | jq -r '.updated_at // "N/A"')
        local description=$(echo "$response" | jq -r '.description // "N/A"')

        log_verbose "  Stars: $stars"
        log_verbose "  Forks: $forks"
        log_verbose "  Open Issues: $open_issues"

        cat <<EOF
{
  "repository": "$repo",
  "stars": $stars,
  "forks": $forks,
  "open_issues": $open_issues,
  "watchers": $watchers,
  "created_at": "$created_at",
  "updated_at": "$updated_at",
  "description": "$description",
  "url": "https://github.com/$repo"
}
EOF
    else
        cat <<EOF
{
  "repository": "$repo",
  "url": "https://github.com/$repo",
  "note": "Install jq for detailed metrics"
}
EOF
    fi
}

################################################################################
# CI/CD Status Collection
################################################################################

get_ci_status() {
    local repo=$1
    log_info "Collecting CI/CD status for $repo..."

    if command -v gh &> /dev/null; then
        local runs
        runs=$(gh run list --repo "$repo" --limit 5 --json status,conclusion,name,createdAt 2>/dev/null || echo "[]")

        if [ "$runs" != "[]" ]; then
            log_verbose "  Latest CI runs fetched"
            echo "$runs"
            return 0
        fi
    fi

    log_warning "Could not fetch CI/CD status (gh CLI required)"
    echo "[]"
}

################################################################################
# Main Collection Logic
################################################################################

collect_all_metrics() {
    log_info "Starting metrics collection for DeepBridge ecosystem..."
    echo ""

    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Start JSON output
    echo "{"
    echo "  \"collection_timestamp\": \"$timestamp\","
    echo "  \"pypi_packages\": ["

    # Collect PyPI metrics
    for i in "${!PACKAGES[@]}"; do
        local package="${PACKAGES[$i]}"
        get_pypi_metrics "$package"
        if [ $i -lt $((${#PACKAGES[@]} - 1)) ]; then
            echo ","
        fi
    done

    echo "  ],"
    echo "  \"github_repositories\": ["

    # Collect GitHub metrics
    for i in "${!GITHUB_REPOS[@]}"; do
        local repo="${GITHUB_REPOS[$i]}"
        get_github_metrics "$repo"
        if [ $i -lt $((${#GITHUB_REPOS[@]} - 1)) ]; then
            echo ","
        fi
    done

    echo "  ],"
    echo "  \"ci_status\": ["

    # Collect CI/CD status
    for i in "${!GITHUB_REPOS[@]}"; do
        local repo="${GITHUB_REPOS[$i]}"
        echo "    {"
        echo "      \"repository\": \"$repo\","
        echo "      \"runs\": $(get_ci_status "$repo")"
        echo "    }"
        if [ $i -lt $((${#GITHUB_REPOS[@]} - 1)) ]; then
            echo ","
        fi
    done

    echo "  ]"
    echo "}"
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --verbose|-v)
                VERBOSE=1
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--output FILE] [--verbose]"
                echo ""
                echo "Options:"
                echo "  --output FILE    Output file path (default: metrics_TIMESTAMP.json)"
                echo "  --verbose, -v    Enable verbose output"
                echo "  --help, -h       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check dependencies
    if ! check_dependencies; then
        log_error "Missing required dependencies"
        exit 1
    fi

    # Collect metrics
    log_info "Output will be saved to: $OUTPUT_FILE"
    echo ""

    collect_all_metrics > "$OUTPUT_FILE"

    echo ""
    log_success "Metrics collection complete!"
    log_info "Results saved to: $OUTPUT_FILE"

    # Pretty print if jq is available
    if command -v jq &> /dev/null && [ "$VERBOSE" -eq 1 ]; then
        echo ""
        log_info "Summary:"
        jq '.' "$OUTPUT_FILE" || cat "$OUTPUT_FILE"
    fi
}

# Run main function
main "$@"
