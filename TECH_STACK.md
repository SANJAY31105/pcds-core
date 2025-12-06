# PCDS Enterprise - Complete Technology Stack

## 🏗️ System Architecture Overview

**Type**: Full-stack Web Application  
**Architecture**: Client-Server with RESTful APIs  
**Deployment**: Docker-ready, Production-grade

---

## 🎯 Backend Stack

### Core Framework
- **FastAPI** (0.104+)
  - Modern Python web framework
  - Async/await support
  - Automatic API documentation (Swagger/OpenAPI)
  - High performance (comparable to Node.js)
  - Built-in validation with Pydantic

### Language
- **Python** 3.11+
  - Type hints for code quality
  - Async programming capabilities
  - Rich ecosystem for AI/ML

### Database
- **SQLite** 3.x
  - Embedded database (no separate server)
  - ACID compliant
  - Perfect for 100K-1M records
  - Zero configuration
  - File-based (easy backups)

**ORM**: SQLAlchemy 2.x
- Object-Relational Mapping
- Database migrations
- Query optimization
- Connection pooling

### AI/ML Stack
- **PyTorch** 2.x
  - Deep learning framework
  - LSTM models for anomaly detection
  - Production-ready inference
  - GPU support (optional)

- **scikit-learn** 1.3+
  - Classical ML algorithms
  - Feature engineering
  - Model evaluation

- **NumPy** / **pandas**
  - Data manipulation
  - Statistical analysis
  - Time series processing

### API & Networking
- **Uvicorn** (ASGI server)
  - High-performance async server
  - WebSocket support
  - Production-ready

- **python-requests**
  - HTTP client library
  - API integrations

- **CORS Middleware**
  - Cross-origin resource sharing
  - Secure frontend-backend communication

### Validation & Data Models
- **Pydantic** 2.x
  - Data validation
  - Settings management
  - Type safety
  - Automatic error handling

### Security
- **python-dotenv**
  - Environment variable management
  - Secure configuration

- **Passlib** (if auth enabled)
  - Password hashing
  - Bcrypt algorithm

### Utilities
- **python-dateutil**
  - Date/time manipulation
  - Timezone handling

---

## 💻 Frontend Stack

### Core Framework
- **Next.js** 14.x
  - React framework
  - Server-side rendering (SSR)
  - Static site generation (SSG)
  - API routes
  - File-based routing
  - Image optimization

### Language
- **TypeScript** 5.x
  - Type-safe JavaScript
  - Enhanced IDE support
  - Better code quality
  - Compile-time error checking

### UI Framework
- **React** 18.x
  - Component-based architecture
  - Hooks for state management
  - Virtual DOM
  - Concurrent rendering

### Styling
- **Tailwind CSS** 3.x
  - Utility-first CSS framework
  - Responsive design
  - Custom color system
  - JIT compiler
  - Minimal bundle size

### UI Components
- **shadcn/ui**
  - Accessible components
  - Headless UI primitives
  - Radix UI based
  - Customizable design

- **Lucide React**
  - Icon library (800+ icons)
  - Tree-shakeable
  - Consistent design

### Data Visualization
- **Recharts** 2.x
  - React-based charts
  - Bar, Line, Pie, Area charts
  - Responsive
  - Customizable

### State Management
- **React Hooks**
  - useState, useEffect
  - Custom hooks
  - Context API (if needed)

### HTTP Client
- **Fetch API** (native)
  - Modern browser API
  - Promise-based
  - No external dependencies

### Build Tools
- **Webpack** 5 (via Next.js)
  - Module bundling
  - Code splitting
  - Tree shaking
  - Hot module replacement

- **PostCSS**
  - CSS processing
  - Autoprefixer
  - CSS optimization

- **Babel** (via Next.js)
  - JavaScript transpilation
  - Modern syntax support

---

## 🗄️ Database Schema

### Tables
1. **detections**
   - Primary threat detection records
   - 100,054 records (in demo)
   - Indexed columns: severity, entity_id, technique_id

2. **entities**
   - Tracked network entities
   - 599 records (in demo)
   - AI-computed threat scores

3. **campaigns**
   - Correlated attack campaigns
   - 8 active campaigns
   - Multi-detection tracking

4. **mitre_techniques** (optional)
   - MITRE ATT&CK framework data
   - Technique mappings

5. **investigations** (optional)
   - Incident investigation cases
   - Timeline tracking

### Indexes (Performance)
- 10 strategic indexes
- Query time: 100ms → 2-4ms
- Covering indexes for common queries

---

## 🔧 Development Tools

### Backend
- **pytest** - Unit testing
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking

### Frontend
- **ESLint** - JavaScript linting
- **Prettier** - Code formatting
- **TypeScript Compiler** - Type checking

### Version Control
- **Git** - Source control
- **GitHub/GitLab** - Repository hosting

---

## 🐳 DevOps & Deployment

### Containerization
- **Docker** 24.x
  - Application containerization
  - Isolated environments
  - Reproducible builds

- **Docker Compose** 2.x
  - Multi-container orchestration
  - Service definition
  - Network configuration

### Web Server (Production)
- **Nginx** (optional)
  - Reverse proxy
  - Load balancing
  - SSL termination
  - Static file serving

### SSL/TLS
- **Let's Encrypt**
  - Free SSL certificates
  - Automatic renewal

### Monitoring (Optional)
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **Sentry** - Error tracking

---

## 🔒 Security Stack

### Authentication (if enabled)
- JWT tokens
- Bcrypt password hashing
- Session management

### Security Headers
- CORS configuration
- CSP (Content Security Policy)
- HSTS (HTTP Strict Transport Security)

### Input Validation
- Pydantic models (backend)
- TypeScript types (frontend)
- SQL injection prevention
- XSS protection

---

## 📦 Key Dependencies

### Backend (requirements.txt)
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
pydantic==2.5.0
python-dotenv==1.0.0
torch==2.1.0
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.3
requests==2.31.0
python-dateutil==2.8.2
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "next": "14.0.3",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "typescript": "5.3.2",
    "tailwindcss": "3.3.5",
    "lucide-react": "0.292.0",
    "recharts": "2.10.3"
  }
}
```

---

## 🌐 API Architecture

### RESTful API Design
- **Base URL**: `/api/v2`
- **Format**: JSON
- **Methods**: GET, POST, PUT, DELETE
- **Status Codes**: Standard HTTP

### Endpoints
```
GET  /api/v2/dashboard/overview
GET  /api/v2/entities
GET  /api/v2/entities/{id}
GET  /api/v2/detections
GET  /api/v2/reports/executive-summary
GET  /api/v2/reports/threat-intelligence
GET  /api/v2/mitre/heatmap
POST /api/v2/investigations
```

### Real-time Updates
- Polling mechanism (5-second intervals)
- WebSocket support (optional)

---

## 💾 Data Flow

### Request Flow
```
User → Next.js Frontend → Fetch API → 
FastAPI Backend → SQLAlchemy → SQLite → 
Response (JSON) → React Components → UI
```

### AI Processing Flow
```
Detection Data → Feature Extraction → 
PyTorch LSTM Model → Anomaly Score → 
Entity Risk Scoring → Database Update → 
Frontend Display
```

---

## 📊 Performance Stack

### Optimization Techniques
- **Database**: 10 strategic indexes
- **Frontend**: Code splitting, lazy loading
- **Backend**: Async operations, connection pooling
- **Caching**: In-memory caching (optional)
- **Compression**: Gzip compression

### Benchmarks
- API Response: 2-4ms average
- Page Load: < 1 second
- Database Queries: < 10ms
- Handles: 100,000+ records

---

## 🧪 Testing Stack

### Backend Testing
- **pytest** - Test framework
- **pytest-asyncio** - Async test support
- **httpx** - HTTP testing client

### Frontend Testing (Optional)
- **Jest** - Unit testing
- **React Testing Library** - Component testing
- **Playwright** - E2E testing

---

## 📱 Cross-Platform Support

### Browser Compatibility
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Responsive Design
- Mobile (320px+)
- Tablet (768px+)
- Desktop (1024px+)
- Large screens (1920px+)

---

## 🔄 CI/CD (Optional)

### Pipeline Tools
- **GitHub Actions** - Automated workflows
- **GitLab CI** - Continuous integration
- **Docker Hub** - Container registry

### Automation
- Automated testing
- Code quality checks
- Docker image builds
- Deployment automation

---

## 🎨 Design System

### Color Palette
- **Severity Colors**:
  - Critical: Red (#EF4444)
  - High: Orange (#F97316)
  - Medium: Yellow (#EAB308)
  - Low: Blue (#3B82F6)

- **Brand Colors**:
  - Primary: Cyan (#06B6D4)
  - Secondary: Purple (#A855F7)
  - Accent: Pink (#EC4899)

### Typography
- Font Family: Inter (from Google Fonts)
- Fallbacks: System fonts

### Animations
- CSS transitions
- Framer Motion (optional)
- Smooth scroll

---

## 🌟 Key Features Enabled by Stack

### Scalability
- Async architecture (FastAPI + Uvicorn)
- Efficient database queries (SQLAlchemy + indexes)
- Code splitting (Next.js)

### Performance
- Server-side rendering (Next.js)
- API route caching
- Optimized database schema

### Developer Experience
- Type safety (TypeScript + Pydantic)
- Auto-generated API docs (FastAPI)
- Hot reload (Next.js + Uvicorn)

### Production Ready
- Docker containerization
- Environment-based configuration
- Comprehensive error handling
- Security best practices

---

## 📈 Scalability Considerations

### Current Limits
- SQLite: 100K-1M records (tested)
- Concurrent users: 50-100
- Storage: File-based database

### Scale-Up Path
- **Database**: Migrate to PostgreSQL
- **Caching**: Add Redis layer
- **Load Balancing**: Nginx + multiple backends
- **CDN**: Static asset delivery
- **Database Sharding**: For 10M+ records

---

## 🎯 Technology Choices - Why?

### FastAPI
- **Why**: Modern, fast, auto-docs, async support
- **Alternative**: Flask, Django

### Next.js
- **Why**: SSR, best React framework, great DX
- **Alternative**: Create React App, Vite

### TypeScript
- **Why**: Type safety, better maintainability
- **Alternative**: JavaScript

### Tailwind CSS
- **Why**: Utility-first, fast development, consistent
- **Alternative**: Bootstrap, Material-UI

### SQLite
- **Why**: Zero-config, perfect for demo, easy deployment
- **Alternative**: PostgreSQL (for production scale)

### PyTorch
- **Why**: Industry-standard for deep learning
- **Alternative**: TensorFlow, scikit-learn

---

## 🚀 Total Tech Count

**Languages**: 4 (Python, TypeScript, JavaScript, SQL)  
**Frameworks**: 2 major (FastAPI, Next.js)  
**Libraries**: 20+ (frontend + backend)  
**Tools**: 10+ (Docker, Git, etc.)  
**Total Packages**: 30+ dependencies

---

**Modern, Production-Grade, Enterprise-Ready Stack!** 🎉
