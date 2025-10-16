#!/bin/bash

# ML Voice Lead Analysis - Multi-Platform Deployment Script
# Automated deployment to Vercel, Render, Railway, and Docker

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

check_requirements() {
    log_info "Checking deployment requirements..."
    command -v git >/dev/null 2>&1 || { log_error "Git required"; exit 1; }
    log_success "Requirements check completed"
}

deploy_vercel() {
    log_info "Deploying to Vercel..."
    
    if ! command -v vercel >/dev/null 2>&1; then
        npm install -g vercel
    fi
    
    vercel --prod --confirm
    log_success "Vercel deployment completed"
}

run_tests() {
    log_info "Running tests..."
    cd backend && python -m pytest tests/ -v || exit 1
    cd .. && cd frontend && npm test -- --watchAll=false || exit 1
    cd ..
    log_success "All tests passed"
}

# Main execution
ENVIRONMENT=${1:-development}
TARGET=${2:-local}

log_info "Deploying to $TARGET (environment: $ENVIRONMENT)"

check_requirements

case $TARGET in
    vercel)
        deploy_vercel
        ;;
    local)
        log_info "Setting up local environment..."
        cd backend && pip install -r requirements.txt && cd ..
        cd frontend && npm install && cd ..
        log_success "Local setup completed"
        ;;
    *)
        log_error "Unsupported target: $TARGET"
        exit 1
        ;;
esac

log_success "Deployment completed successfully!"