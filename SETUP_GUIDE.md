# ML Voice Lead Analysis - Complete Setup Guide

This guide provides step-by-step instructions for setting up the ML Voice Lead Analysis platform locally and deploying to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)
6. [Production Deployment](#production-deployment)

## Prerequisites

### Required Software

- **Docker Desktop** (recommended)
  - Download: https://www.docker.com/products/docker-desktop
  - Version: 20.10 or higher
  - Includes Docker Compose

- **Alternative: Manual Setup**
  - Python 3.11 or higher
  - Node.js 18.0 or higher
  - PostgreSQL 15 or higher
  - Redis 7 or higher

### AWS Account (Optional for Local Development)

- AWS account with S3 bucket access
- IAM user with appropriate permissions
- Access key and secret key

For local development without AWS, you can disable AWS checks (see Configuration section).

## Local Development Setup

### Option 1: Docker Compose (Recommended)

This is the easiest way to get started. All services will run in containers.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/aaron-seq/ML-voice-lead-analysis.git
cd ML-voice-lead-analysis
```

#### Step 2: Create Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your settings
# For local development, you can use the default values
# or set DISABLE_AWS_CHECKS=true to skip AWS requirements
```

Minimal `.env` for local development:

```env
ENVIRONMENT=development
DEBUG=true
DISABLE_AWS_CHECKS=true

# Database
POSTGRES_DB=voice_analysis
POSTGRES_USER=voice_user
POSTGRES_PASSWORD=local_dev_password

# Redis (use defaults)
REDIS_URL=redis://redis:6379/0

# Security (generate secure keys for production)
SECRET_KEY=local-development-secret-key-change-for-production
```

#### Step 3: Start All Services

```bash
# Build and start all containers
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

This will start:
- Backend API (http://localhost:8000)
- Frontend Dashboard (http://localhost:3000)
- PostgreSQL Database (localhost:5432)
- Redis Cache (localhost:6379)

#### Step 4: Verify Services

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Test backend health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/v1/docs
```

### Option 2: Manual Setup

For developers who prefer not to use Docker.

#### Step 1: Install Dependencies

**Backend Setup:**

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Install ML models
python scripts/install_models.py
```

**Frontend Setup:**

```bash
cd frontend

# Install Node packages
npm install

# Or use yarn
yarn install
```

#### Step 2: Set Up Database

**PostgreSQL:**

```bash
# Install PostgreSQL (if not already installed)
# macOS: brew install postgresql@15
# Ubuntu: sudo apt-get install postgresql-15

# Start PostgreSQL service
# macOS: brew services start postgresql@15
# Ubuntu: sudo systemctl start postgresql

# Create database and user
psql postgres
```

```sql
CREATE DATABASE voice_analysis;
CREATE USER voice_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE voice_analysis TO voice_user;
\q
```

```bash
# Run database initialization
psql -U voice_user -d voice_analysis -f database/init/01_init_schema.sql
```

**Redis:**

```bash
# Install Redis (if not already installed)
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server

# Start Redis service
# macOS: brew services start redis
# Ubuntu: sudo systemctl start redis

# Test Redis connection
redis-cli ping
# Should return: PONG
```

#### Step 3: Configure Environment

Create `.env` file in project root:

```env
ENVIRONMENT=development
DEBUG=true
DISABLE_AWS_CHECKS=true

DATABASE_URL=postgresql+asyncpg://voice_user:your_password@localhost:5432/voice_analysis
REDIS_URL=redis://localhost:6379/0

SECRET_KEY=your-secret-key-here
```

#### Step 4: Start Services

**Terminal 1 - Backend:**

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm run dev
# or: yarn dev
```

## Configuration

### Environment Variables

See `.env.example` for complete list of configuration options.

#### Key Configuration Options

**Application Mode:**

```env
ENVIRONMENT=development        # development, staging, or production
DEBUG=true                     # Enable debug mode
DISABLE_AWS_CHECKS=true        # Skip AWS connectivity checks
```

**AWS Configuration:**

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
DATA_BUCKET=your-bucket-name
```

**Database:**

```env
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/dbname

# SQLite (alternative for local dev)
DATABASE_URL=sqlite+aiosqlite:///./voice_analysis.db
```

**ML Models:**

```env
SPACY_MODEL=en_core_web_md    # or en_core_web_sm, en_core_web_lg
USE_TRANSFORMERS=false         # Enable transformer models (requires GPU)
```

### Manual ML Model Installation

If automatic installation fails:

```bash
cd backend
source venv/bin/activate

# Install spaCy models
python -m spacy download en_core_web_md
# or smaller model: python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"

# Verify installation
python scripts/install_models.py --verify
```

## Running the Application

### Development Mode

**With Docker Compose:**

```bash
# Start services
docker-compose up

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build

# View logs
docker-compose logs -f
```

**Manual Mode:**

```bash
# Backend (terminal 1)
cd backend && uvicorn main:app --reload

# Frontend (terminal 2)
cd frontend && npm run dev
```

### Accessing the Application

- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/v1/docs
- **Alternative Docs:** http://localhost:8000/v1/redoc
- **Health Check:** http://localhost:8000/health

### Testing the API

**Using curl:**

```bash
# Health check
curl http://localhost:8000/health

# List calls (with mock data)
curl http://localhost:8000/v1/calls?page=1&page_size=10

# Get specific call analysis
curl http://localhost:8000/v1/calls/sample-call-001.json
```

**Using the Swagger UI:**

1. Open http://localhost:8000/v1/docs
2. Expand any endpoint
3. Click "Try it out"
4. Enter parameters
5. Click "Execute"

## Troubleshooting

### Common Issues

#### Docker Issues

**Issue: Port already in use**

```bash
# Check what's using the port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Solution: Stop the conflicting service or change port in .env
BACKEND_PORT=8001
```

**Issue: Container fails to start**

```bash
# Check logs
docker-compose logs backend

# Rebuild from scratch
docker-compose down -v
docker-compose up --build
```

#### Database Issues

**Issue: Cannot connect to PostgreSQL**

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection string in .env
DATABASE_URL=postgresql+asyncpg://voice_user:password@postgres:5432/voice_analysis
```

**Issue: Database schema not initialized**

```bash
# Manually run initialization script
docker-compose exec postgres psql -U voice_user -d voice_analysis -f /docker-entrypoint-initdb.d/01_init_schema.sql
```

#### Redis Issues

**Issue: Redis connection failed**

```bash
# Check if Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping
# Should return: PONG

# Check logs
docker-compose logs redis
```

#### ML Model Issues

**Issue: spaCy model not found**

```bash
# Install manually
docker-compose exec backend python scripts/install_models.py

# Or with pip
docker-compose exec backend python -m spacy download en_core_web_md

# Verify installation
docker-compose exec backend python scripts/install_models.py --verify
```

#### AWS Issues

**Issue: AWS credentials not configured**

```bash
# For local development, disable AWS checks
DISABLE_AWS_CHECKS=true

# Or configure AWS credentials
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### Debug Mode

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG

# Check service health
curl http://localhost:8000/health

# View detailed logs
docker-compose logs -f backend
```

## Production Deployment

### Pre-Deployment Checklist

- [ ] Update `.env` with production values
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Generate secure `SECRET_KEY` (min 32 characters)
- [ ] Configure AWS credentials
- [ ] Set up SSL certificates
- [ ] Configure database backups
- [ ] Set up monitoring and logging

### Security Configuration

```env
# Production environment
ENVIRONMENT=production
DEBUG=false

# Secure secrets (generate with: openssl rand -hex 32)
SECRET_KEY=your-cryptographically-secure-secret-key-here
JWT_SECRET_KEY=another-secure-secret-for-jwt-tokens

# Database with SSL
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db?ssl=require

# Redis with password
REDIS_URL=redis://:secure_password@redis:6379/0

# Restrict CORS origins
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
```

### Deployment Platforms

#### Docker Production

```bash
# Build production images
docker-compose -f docker-compose.yml build

# Start with production profile
ENVIRONMENT=production docker-compose up -d
```

#### Cloud Platforms

- **Vercel:** Follow deployment guide in README
- **Railway:** Push to GitHub and connect repository
- **Render:** Use `render.yaml` configuration
- **AWS:** Use provided Lambda/ECS configurations

For detailed cloud deployment instructions, see README.md.

## Additional Resources

- **Main Documentation:** README.md
- **API Documentation:** http://localhost:8000/v1/docs
- **GitHub Issues:** https://github.com/aaron-seq/ML-voice-lead-analysis/issues
- **GitHub Discussions:** https://github.com/aaron-seq/ML-voice-lead-analysis/discussions

## Getting Help

If you encounter issues not covered in this guide:

1. Check existing GitHub issues
2. Review application logs
3. Enable debug mode for detailed error messages
4. Create a new issue with:
   - Error messages
   - Steps to reproduce
   - Environment details
   - Relevant log outputs

---

**Happy coding! If you have suggestions to improve this guide, please submit a pull request.**
