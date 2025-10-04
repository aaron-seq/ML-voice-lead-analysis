
# ML Voice Lead Analysis Pipeline

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/aaronseq12/ML-voice-lead-analysis)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![React](https://img.shields.io/badge/react-18.2.0-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Next-Generation AI-Powered Sales Call Analysis System**
> 
> Transform your sales conversations into actionable insights with advanced ML, real-time sentiment analysis, and intelligent lead scoring.

## Features

### **Advanced AI Analysis**
- **Multi-Model Sentiment Analysis** - VADER, TextBlob, and Transformer-based ensemble
- **Intelligent Lead Scoring** - AI-powered classification with confidence metrics
- **Topic Extraction** - Automatic identification of discussion themes
- **Excitement Detection** - "Wow moments" and high-interest signals
- **Speaker Profiling** - Role identification and engagement analysis

### **Rich Dashboard**
- **Real-time Analytics** - Live processing status and metrics
- **Interactive Visualizations** - Charts, trends, and heatmaps
- **Advanced Filtering** - Search, sort, and filter calls efficiently
- **Export Capabilities** - PDF reports and data export
- **Mobile Responsive** - Optimized for all devices

### **Performance & Scalability**
- **Async Processing** - High-throughput pipeline architecture
- **Cloud-Native** - AWS S3, Transcribe, and Lambda integration
- **Caching Layer** - Redis for optimal performance
- **Docker Support** - Containerized deployment
- **Production Ready** - Health checks, monitoring, and logging

## Prerequisites

### System Requirements
- **Python 3.9+**
- **Node.js 18.0+**
- **Docker & Docker Compose**
- **Git**

### Required Accounts
- **AWS Account** (for S3 and Transcribe services)
- **OpenAI API Key** (optional, for enhanced analysis)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aaronseq12/ML-voice-lead-analysis.git
cd ML-voice-lead-analysis
```

### 2. Environment Setup

```bash
# Copy environment templates
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Edit the environment files with your configurations
```

### 3. Install Dependencies

#### Option A: Docker (Recommended)
```bash
# Start all services with Docker Compose
docker-compose up --build

# The application will be available at:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/v1/docs
```

#### Option B: Manual Installation
```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Frontend setup (in new terminal)
cd frontend
npm install

# Start services
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### 4. Verify Installation

Visit these URLs to confirm everything is working:

-  **Frontend Dashboard**: http://localhost:3000
-  **API Documentation**: http://localhost:8000/v1/docs  
-  **Health Check**: http://localhost:8000/health

## Configuration

### Backend Configuration (`backend/.env`)

```bash
# Application Settings
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# AWS Configuration
DATA_BUCKET=your-s3-bucket-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Database
DATABASE_URL=postgresql+asyncpg://voice_user:voice_pass@localhost:5432/voice_analysis

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-in-production
```

### Frontend Configuration (`frontend/.env.local`)

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=ML Voice Lead Analysis
NEXT_PUBLIC_VERSION=3.0.0

# Environment
NODE_ENV=development
```

## Usage Guide

### Processing Your First Call

1. **Upload Audio File** to your configured S3 bucket under `transcripts/` prefix
2. **Run Analysis Pipeline**:
   ```bash
   cd pipeline
   python enhanced_analysis_pipeline.py transcripts/your-call.json
   ```
3. **View Results** in the dashboard at http://localhost:3000

### API Endpoints

#### Core Endpoints
- `GET /v1/calls` - List all analyzed calls (paginated)
- `GET /v1/calls/{file_name}` - Get detailed analysis for specific call
- `POST /v1/calls/{file_name}/reanalyze` - Trigger re-analysis
- `GET /health` - System health check

#### Example API Usage

```python
import requests

# Get list of calls
response = requests.get("http://localhost:8000/v1/calls?page=1&page_size=10")
calls = response.json()

# Get detailed analysis
response = requests.get("http://localhost:8000/v1/calls/sample-call.json")
analysis = response.json()

print(f"Lead Score: {analysis['leadScore']['primary_score']}")
print(f"Sentiment: {analysis['sentiment_analysis']['overall_score']}")
```

## 🧪 Testing

### Run All Tests

```bash
# Backend tests
cd backend
pytest tests/ -v --coverage

# Frontend tests  
cd frontend
npm test

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Test Coverage

The project maintains >90% test coverage across:
- API endpoint testing
- ML pipeline validation
- Frontend component testing
- Integration testing

## 📊 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │  ML Pipeline    │
│   (React/TS)    │◄──►│   (FastAPI)     │◄──►│   (Python)      │
│                 │    │                 │    │                 │
│ • Dashboard     │    │ • REST API      │    │ • NLP Processing│
│ • Visualizations│    │ • Data Models   │    │ • Lead Scoring  │
│ • User Interface│    │ • Business Logic│    │ • Sentiment Anal│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                ┌─────────────────▼─────────────────┐
                │           Data Layer              │
                │                                   │
                │  ┌─────────┐ ┌─────────┐ ┌──────┐ │
                │  │   S3    │ │PostgreSQL│ │Redis │ │
                │  │(Storage)│ │(Database) │ │(Cache)│ │
                │  └─────────┘ └─────────┘ └──────┘ │
                └───────────────────────────────────┘
```

### Key Components

- **Frontend**: Modern React with TypeScript, Tailwind CSS, and Framer Motion
- **Backend**: FastAPI with async processing, SQLAlchemy ORM, and Pydantic validation
- **ML Pipeline**: spaCy, TensorFlow, Transformers, and custom algorithms
- **Storage**: AWS S3 for files, PostgreSQL for structured data, Redis for caching
- **Infrastructure**: Docker containers, Nginx reverse proxy, monitoring stack

## Deployment

### Development Deployment
```bash
# Start development environment
docker-compose up --build

# Or start individual services
docker-compose up postgres redis -d  # Start dependencies
npm run dev                         # Frontend development server
uvicorn main:app --reload           # Backend development server
```

### Production Deployment

#### Docker Production
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d
```

#### Vercel Deployment (Frontend)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

#### AWS Lambda (Backend)
```bash
# Package for Lambda deployment
cd backend
pip install -r requirements.txt -t lambda_package/
cp -r app/ lambda_package/
cd lambda_package && zip -r ../lambda_deployment.zip .

# Deploy using AWS CLI or Console
aws lambda update-function-code \
  --function-name ml-voice-analysis \
  --zip-file fileb://lambda_deployment.zip
```

### Environment-Specific Configurations

#### Staging
- Reduced resource allocation
- Test data integration
- Performance monitoring

#### Production  
- Auto-scaling enabled
- Full monitoring stack
- Backup strategies
- Security hardening

## Performance Optimization

### Backend Optimizations
- **Async Processing**: All I/O operations use async/await
- **Connection Pooling**: Optimized database and Redis connections
- **Caching Strategy**: Multi-layer caching with Redis
- **Request Batching**: Bulk operations for efficiency

### Frontend Optimizations
- **Code Splitting**: Lazy loading of components
- **Image Optimization**: Next.js automatic optimization
- **Bundle Analysis**: Webpack bundle analyzer integration
- **Performance Monitoring**: Web vitals tracking

### ML Pipeline Optimizations
- **Model Caching**: Cached embeddings and model outputs
- **GPU Acceleration**: CUDA support for TensorFlow operations
- **Batch Processing**: Vectorized operations for speed
- **Memory Management**: Efficient memory usage patterns

## 🔐 Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting
- CORS configuration

### Data Protection
- Encryption at rest (S3, Database)
- Encryption in transit (HTTPS/TLS)
- Input validation and sanitization
- SQL injection prevention

### Infrastructure Security
- Container security scanning
- Dependency vulnerability checks
- Network security groups
- Environment variable protection

## 📊 Monitoring & Observability

### Metrics Collection
- **Application Metrics**: Response times, error rates, throughput
- **Business Metrics**: Processing success rate, analysis accuracy
- **Infrastructure Metrics**: CPU, memory, disk usage

### Logging Strategy
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: DEBUG, INFO, WARN, ERROR with proper categorization  
- **Log Aggregation**: Centralized logging with ELK stack

### Health Monitoring
- **Health Checks**: Deep health checks for all components
- **Alerting**: PagerDuty integration for critical issues
- **Dashboards**: Grafana dashboards for visualization

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests
4. **Run the test suite**: `npm test` and `pytest`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

### Code Standards
- **Python**: Black formatting, flake8 linting, type hints
- **TypeScript**: Prettier formatting, ESLint, strict type checking
- **Commit Messages**: Conventional Commits format
- **Documentation**: Comprehensive docstrings and comments

## 🔧 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check Python version
python --version  # Should be 3.9+

# Verify dependencies
pip install -r requirements.txt

# Check environment variables
cat backend/.env

# Verify spaCy model
python -m spacy validate
```

#### Frontend Build Fails
```bash
# Clear cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+

# Verify environment variables
cat frontend/.env.local
```

#### ML Pipeline Errors
```bash
# Download required models
python -m spacy download en_core_web_lg

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Verify NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### AWS Connection Issues
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://your-bucket-name

# Check IAM permissions
aws iam get-user
```

### Performance Issues

#### Slow API Response
- Check database query performance
- Verify Redis cache hit rates
- Monitor CPU and memory usage
- Review slow query logs

#### High Memory Usage
- Adjust ML model batch sizes
- Implement model unloading
- Optimize database connection pools
- Monitor garbage collection

### Getting Help

- 📖 **Documentation**: Check our [comprehensive docs](docs/)
- 🐛 **Bug Reports**: [Create an issue](https://github.com/aaronseq12/ML-voice-lead-analysis/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/aaronseq12/ML-voice-lead-analysis/discussions)
- 📧 **Email**: aaronsequeira12@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy Team** for excellent NLP libraries
- **FastAPI** for the modern Python web framework
- **React Team** for the powerful frontend framework  
- **TensorFlow** for ML infrastructure
- **Open Source Community** for incredible tools and libraries

---

## Project Statistics

- **Lines of Code**: 15,000+
- **Test Coverage**: 90%+
- **API Endpoints**: 20+
- **ML Models**: 5+ integrated
- **Supported Languages**: English (extensible)
- **Processing Speed**: ~2-3 minutes per call
- **Accuracy**: 85%+ lead scoring accuracy

---

<div align="center">

**Built with ❤️ by [Aaron Sequeira](https://github.com/aaronseq12)**

[⭐ Star this repo](https://github.com/aaronseq12/ML-voice-lead-analysis) | [🐛 Report Bug](https://github.com/aaronseq12/ML-voice-lead-analysis/issues) | [✨ Request Feature](https://github.com/aaronseq12/ML-voice-lead-analysis/issues)

</div>
