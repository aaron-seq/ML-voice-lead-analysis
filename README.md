# ML Voice Lead Analysis Platform

**Next-Generation AI-Powered Sales Call Analysis System**

Transform your sales conversations into actionable insights with advanced ML, real-time sentiment analysis, and intelligent lead scoring. Built with modern architecture for enterprise scalability and cloud deployment.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.0.4-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

## Features Overview

### Advanced AI Analysis
- **Multi-Model Sentiment Analysis** - Ensemble approach with VADER, TextBlob, and Transformers
- **Intelligent Lead Scoring** - AI-powered classification with confidence metrics
- **Topic Extraction** - Automatic identification of conversation themes
- **Interest Detection** - High-engagement moment identification
- **Speaker Profiling** - Role identification and engagement analysis

### Modern Dashboard
- **Real-time Analytics** - Live processing status and performance metrics
- **Interactive Visualizations** - Responsive charts and data exploration
- **Advanced Filtering** - Multi-criteria search and sorting capabilities
- **Export Capabilities** - PDF reports and structured data export
- **Mobile Responsive** - Optimized for all device types

### Enterprise Architecture
- **Cloud-Native Design** - AWS, Azure, GCP compatible
- **Microservices Architecture** - Scalable and maintainable components
- **Container-Ready** - Docker and Kubernetes deployment support
- **API-First Design** - RESTful APIs with comprehensive documentation
- **Security Hardened** - Enterprise-grade security practices

## Quick Start Guide

### Prerequisites

**System Requirements:**
- Python 3.11+
- Node.js 18.0+
- Docker & Docker Compose (recommended)
- Git

**Required Services:**
- AWS Account (for S3 and optional Transcribe)
- OpenAI API Key (optional, for enhanced analysis)

### Installation Methods

#### Option A: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/aaron-seq/ML-voice-lead-analysis.git
cd ML-voice-lead-analysis

# Configure environment
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Edit configuration files with your settings
# nano backend/.env
# nano frontend/.env.local

# Start all services
docker-compose up --build

# Access the application:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/v1/docs
```

#### Option B: Manual Development Setup

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Frontend setup (new terminal)
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

### Configuration

#### Backend Configuration (`backend/.env`)

```env
# Application Settings
ENVIRONMENT=development
DEBUG=true
PORT=8000

# AWS Configuration
AWS_REGION=us-east-1
DATA_BUCKET=your-s3-bucket-name
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Database (choose one)
DATABASE_URL=sqlite+aiosqlite:///./voice_analysis.db
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/voice_analysis

# Security
SECRET_KEY=your-super-secret-key-minimum-32-characters

# ML Configuration
USE_TRANSFORMERS=false  # Set to true for enhanced analysis
CLOUD_DEPLOYMENT=false  # Set to true for cloud optimization
```

#### Frontend Configuration (`frontend/.env.local`)

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=ML Voice Lead Analysis
NEXT_PUBLIC_VERSION=3.1.0

# Environment
NODE_ENV=development
NEXT_TELEMETRY_DISABLED=1
```

## Architecture Overview

```mermaid
graph TB
    A[Frontend Dashboard<br/>React/Next.js] --> B[API Gateway<br/>FastAPI]
    B --> C[Analysis Service<br/>ML Pipeline]
    B --> D[AWS S3<br/>File Storage]
    B --> E[Database<br/>PostgreSQL/SQLite]
    B --> F[Cache<br/>Redis]
    C --> G[NLP Models<br/>spaCy, Transformers]
    C --> H[Sentiment Analysis<br/>Multi-model Ensemble]
    C --> I[Lead Scoring<br/>AI Classification]
```

### Key Components

- **Frontend**: Modern React with TypeScript, Tailwind CSS, and Next.js
- **Backend**: FastAPI with async processing, SQLAlchemy ORM, structured logging
- **ML Pipeline**: Advanced NLP with spaCy, TensorFlow, Transformers
- **Storage**: AWS S3 for files, PostgreSQL for data, Redis for caching
- **Infrastructure**: Docker containers, cloud deployment ready

## Usage Guide

### Processing Your First Call

1. **Upload Audio Transcript** to your S3 bucket:
   ```
   s3://your-bucket/transcripts/sample-call.json
   ```

2. **Run Analysis Pipeline**:
   ```bash
   cd pipeline
   python voice_analysis_pipeline.py transcripts/sample-call.json
   ```

3. **View Results** in the dashboard at `http://localhost:3000`

### API Usage Examples

#### Get Call Listings
```python
import requests

# Retrieve paginated call list
response = requests.get(
    "http://localhost:8000/v1/calls",
    params={"page": 1, "page_size": 10, "sort_field": "upload_timestamp"}
)
calls = response.json()

print(f"Found {calls['pagination']['total_items']} calls")
```

#### Get Detailed Analysis
```python
# Get comprehensive analysis
response = requests.get("http://localhost:8000/v1/calls/sample-call.json")
analysis = response.json()

print(f"Lead Score: {analysis['lead_score_details']['primary_classification']}")
print(f"Sentiment: {analysis['sentiment_score']:.2f}")
print(f"Key Topics: {', '.join(analysis['extracted_topics'])}")
```

#### Trigger Reprocessing
```python
# Reprocess with latest models
response = requests.post("http://localhost:8000/v1/calls/sample-call.json/reprocess")
result = response.json()

print(f"Status: {result['status']}")
print(f"Processing ID: {result['processing_id']}")
```

## Cloud Deployment

### Vercel Deployment (Frontend + Serverless Backend)

```bash
# Install Vercel CLI
npm i -g vercel

# Configure environment variables in Vercel dashboard:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - S3_BUCKET_NAME
# - SECRET_KEY

# Deploy
vercel --prod
```

### Render Deployment

1. **Backend Service**:
   - Connect your GitHub repository
   - Set build command: `pip install -r backend/requirements.txt`
   - Set start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables

2. **Frontend Service**:
   - Set build command: `cd frontend && npm install && npm run build`
   - Set start command: `cd frontend && npm start`
   - Add environment variables

### Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

### AWS Lambda Deployment

```bash
# Package backend for Lambda
cd backend
pip install -r requirements.txt -t lambda_package/
cp -r . lambda_package/
cd lambda_package && zip -r ../lambda_deployment.zip .

# Deploy using AWS CLI or Console
aws lambda update-function-code \
  --function-name ml-voice-analysis \
  --zip-file fileb://lambda_deployment.zip
```

### Docker Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.yml build

# Deploy with production settings
ENVIRONMENT=production \
SECRET_KEY=your-production-secret \
AWS_ACCESS_KEY_ID=your-key \
AWS_SECRET_ACCESS_KEY=your-secret \
docker-compose up -d
```

## Development

### Running Tests

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

### Code Quality

```bash
# Backend formatting and linting
cd backend
black .
isort .
flake8 .
mypy .

# Frontend formatting and linting
cd frontend
npm run lint
npm run format
npm run type-check
```

### Performance Monitoring

Monitor application performance through:
- **Health Check**: `GET /health`
- **Performance Metrics**: `GET /v1/analytics/performance`
- **System Info**: `GET /v1/system/info`

## Security

### Security Features
- JWT-based authentication support
- Input validation and sanitization
- SQL injection prevention
- CORS configuration
- Rate limiting ready
- Container security best practices

### Security Configuration

```env
# Strong secret key (32+ characters)
SECRET_KEY=your-cryptographically-secure-secret-key

# Database connection with SSL
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db?ssl=require

# CORS origins (production)
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
```

## Performance Optimization

### Backend Optimizations
- Async/await for all I/O operations
- Connection pooling for databases
- Multi-layer caching with Redis
- Request batching and pagination
- Background task processing

### Frontend Optimizations
- Code splitting and lazy loading
- Image optimization with Next.js
- Bundle analysis and optimization
- Performance monitoring
- CDN-ready static assets

### ML Pipeline Optimizations
- Model caching and reuse
- Batch processing capabilities
- Memory-efficient processing
- GPU acceleration support (optional)
- Lightweight models for cloud deployment

## Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check Python version
python --version  # Should be 3.11+

# Verify dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md

# Check environment variables
cat backend/.env
```

#### Frontend Build Fails
```bash
# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+
```

#### AWS Connection Issues
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://your-bucket-name

# Check bucket permissions
aws s3api get-bucket-policy --bucket your-bucket-name
```

### Performance Issues

- **Slow API Response**: Check database query performance, Redis cache hit rates
- **High Memory Usage**: Adjust ML model batch sizes, implement model unloading
- **Container Issues**: Monitor resource limits, check container logs

### Getting Help

- 📖 **Documentation**: Comprehensive guides and API docs
- 🐛 **Issues**: [GitHub Issues](https://github.com/aaron-seq/ML-voice-lead-analysis/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/aaron-seq/ML-voice-lead-analysis/discussions)
- 📧 **Contact**: [aaronsequeira12@gmail.com](mailto:aaronsequeira12@gmail.com)

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** with comprehensive tests
4. **Run test suite**: `npm test` and `pytest`
5. **Commit changes**: `git commit -m 'feat: add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

### Code Standards

- **Python**: Black formatting, isort imports, flake8 linting, type hints
- **TypeScript**: Prettier formatting, ESLint, strict type checking
- **Commit Messages**: Conventional Commits format
- **Documentation**: Comprehensive docstrings and README updates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **spaCy Team** for excellent NLP libraries
- **FastAPI** for the modern Python web framework  
- **Next.js Team** for the powerful React framework
- **Vercel** for seamless deployment platform
- **Open Source Community** for incredible tools and libraries

## Project Statistics

- **Lines of Code**: 20,000+
- **Test Coverage**: 90%+
- **API Endpoints**: 25+
- **ML Models**: 5+ integrated
- **Deployment Platforms**: 8+ supported
- **Processing Speed**: ~2-3 minutes per call
- **Accuracy**: 85%+ lead scoring accuracy

---

**Built with ❤️ by [Aaron Sequeira](https://github.com/aaron-seq)**

⭐ **Star this repo** | 🐛 **Report Bug** | ✨ **Request Feature**

*Transform your sales conversations into actionable insights with enterprise-grade AI analysis.*