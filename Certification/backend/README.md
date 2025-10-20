# 🎓 School Events Assistant

A full-stack application for searching and querying school events and programs using advanced RAG (Retrieval-Augmented Generation), multi-agent systems, and comprehensive RAGAS evaluation.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Backend](#backend)
- [Frontend](#frontend)
- [Setup](#setup)
- [API Reference](#api-reference)
- [Features](#features)

## 🎯 Overview

The School Events Assistant is an intelligent chatbot application that helps parents and students find information about school programs, events, and activities. It combines:

- **RAG Pipeline**: Dual retrieval methods with vector search
- **Multi-Agent System**: Specialized agents for different information sources
- **RAGAS Evaluation**: Comprehensive metrics for RAG performance
- **Interactive UI**: Modern React frontend with event browsing and chat

## 🏗️ Architecture

```
School Events Assistant
├── Backend (FastAPI + LangChain)
│   ├── RAG Pipeline (Qdrant vector store)
│   ├── Multi-Agent System (LangGraph)
│   ├── Custom Tools (school_events_search)
│   ├── RAGAS Evaluation Framework
│   └── 65 School Events (JSON database)
│
└── Frontend (React)
    ├── Event Browser (Grid view with 65 events)
    ├── Chat Interface (RAG-powered Q&A)
    ├── RAGAS Evaluation Dashboard
    └── Method Comparison Viewer
```

### Data Flow
```
User Query → Frontend → FastAPI → RAG/Agent → LLM → Response → Frontend
                                     ↓
                              Vector Store (Qdrant)
                                     ↓
                              65 Event Documents
```

## 🔧 Backend

### Project Structure

```
backend/
├── app/
│   ├── tools/
│   │   └── school_events_tool.py    # Custom LangChain tool for event search
│   └── agents/
│       └── multi_agent_system.py    # Multi-agent system (WebSearch + LocalEvents)
├── data/                             # 65 school event JSON files
│   ├── coding_boot_camp_kids.json
│   ├── art_classes_kids.json
│   ├── robotics_club.json
│   └── ... (62 more events)
├── generated_results/                # RAGAS evaluation results (JSON)
├── logs/
│   └── backend.log                   # Application logs
├── venv/                             # Python virtual environment
├── main.py                           # FastAPI application (961 lines)
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (not in git)
└── README.md                         # This file
```

### Technology Stack

**Core Framework**
- FastAPI 0.104.1 - Modern web framework
- Uvicorn - ASGI server
- Python 3.10+

**LangChain & AI**
- LangChain 0.2.x - RAG orchestration
- LangGraph 0.2.x - Multi-agent workflows
- LangChain OpenAI - LLM integration
- OpenAI GPT-3.5-turbo - Language model
- text-embedding-3-small - Embeddings

**Vector Database**
- Qdrant (in-memory) - Vector storage and retrieval
- ChromaDB - Tool-specific vector store

**Evaluation**
- RAGAS 0.1.x - RAG evaluation metrics
- Pandas - Data analysis

### RAG Pipeline

#### Dual Retrieval Methods

**1. Original RAG (k=4)**
- Simple prompt chain pattern
- 4 most relevant documents
- Query-focused retrieval
- Faster response time

**2. Naive Retrieval (k=10)**
- LCEL chain pattern (from Advanced Retrieval notebook)
- 10 most relevant documents
- More comprehensive context
- Better for complex queries

#### Document Processing
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ",", " ", ""]
)
```
- **Input**: 65 JSON event files
- **Output**: ~200+ document chunks
- **Embedding**: OpenAI text-embedding-3-small
- **Storage**: Qdrant in-memory vector store

### Multi-Agent System

Built with LangGraph using the Multi-Agent RAG pattern:

**Agent 1: LocalEvents Agent**
- Tool: `school_events_search`
- Purpose: Search local event database
- Vector Store: ChromaDB
- Best for: Event details, schedules, registration

**Agent 2: WebSearch Agent**
- Tool: Tavily Search
- Purpose: Real-time web information
- Best for: Current events, external resources

**Routing Logic**
- Supervisor agent analyzes query
- Routes to appropriate specialized agent
- Combines results when needed

### RAGAS Evaluation

Metrics tracked:
- **Faithfulness**: Factual accuracy (hallucination detection)
- **Answer Relevancy**: Response relevance to question
- **Context Precision**: Quality of retrieved documents
- **Context Recall**: Completeness of retrieved context

Results saved to `generated_results/` directory in JSON format.

## 💻 Frontend

### Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   └── EventPopup.js          # Event details modal
│   ├── assets/
│   │   └── SchoolIcon.svg         # App icon
│   ├── App.js                      # Main application (794 lines)
│   ├── index.css                   # Styles (2290+ lines)
│   └── index.js                    # React entry point
├── package.json
└── README.md
```

### Technology Stack

- **React** 18.2.0 - UI framework
- **Axios** 1.6.0 - HTTP client
- **Material Design Blue** theme (#1976D2, #0D47A1)
- **CSS Grid/Flexbox** - Responsive layouts

### Features

#### 1. Event Browser Tab 🎪
- Grid layout displaying 65 school events
- Event cards with:
  - Icon (🏕️ camps, 🎯 challenges, 🎨 art, etc.)
  - Name and organization
  - Description (truncated to 100 chars)
  - Target audience, date, cost
  - "More Info" button
- Click event card → Auto-fills chat query
- Click "More Info" → Opens detailed popup

#### 2. Chat Interface Tab 💬
- RAG-powered conversational AI
- Features:
  - Markdown formatting (bold **text**)
  - Bullet point lists (• items)
  - Smart response formatting
  - Loading indicators
  - Error handling with connection status
- Message types: User, Assistant, Error
- Auto-scroll to latest message

#### 3. RAGAS Evaluation Tab 📊
- Auto-runs evaluation on tab open
- Displays 4 metrics with:
  - Percentage scores
  - Visual progress bars
  - Metric descriptions
  - Color-coded indicators
- Re-run evaluation button
- Shows test question count

#### 4. Method Comparison Tab 📈
- Side-by-side comparison table
- Compares Original (k=4) vs Naive (k=10)
- Shows improvement deltas (✅/⚠️)
- Visual mini progress bars
- Key insights section
- Auto-runs on first visit

### UI/UX Features

**Event Cards**
- Flexible height (min-height: 280px)
- No text cutoff
- Hover effects
- Smooth animations (staggered 0.1s delay)
- Responsive grid layout

**Event Popup Modal**
- Full event details
- Organized sections (🏢 Organization, 📝 Description)
- Detail grid (👥 Audience, 📅 Date, 💰 Cost, 🏷️ Type)
- "Ask AI About This Event" button
- Click outside to close

**Chat Messages**
- User messages (blue, right-aligned)
- AI responses (gray, left-aligned)
- Icon indicators (👤 user, 🤖 AI)
- Formatted lists and headers
- Typing indicator with animated dots

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Node.js 16+ and npm
- OpenAI API key
- Tavily API key (optional)

### Backend Setup

1. **Navigate to backend directory**
```bash
cd backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create `.env` file:
```env
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here  # Optional
```

5. **Verify data files**
Ensure 65 JSON files exist in `data/` directory

6. **Start server**
```bash
python main.py
```

Server runs on `http://localhost:8000`

**Verify Backend**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/events | head -20
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm start
```

Frontend runs on `http://localhost:3000`

**The app will automatically open in your browser**

### Full Stack Launch

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm start
```

## 📡 API Reference

### Core Endpoints

#### `GET /` - Root endpoint
Health check
```bash
curl http://localhost:8000/
```

#### `GET /health` - Health status
```json
{
  "status": "healthy",
  "timestamp": "2025-10-19T12:00:00"
}
```

#### `POST /query` - Query RAG
Query using active retrieval method (Original or Naive)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What coding programs are available?"}'
```

**Response:**
```json
{
  "answer": "Here are the coding programs available...",
  "context": ["Document chunk 1...", "Document chunk 2..."]
}
```

#### `GET /events` - List events
List all 65 available events
```bash
curl http://localhost:8000/events
```

### Agent Endpoints

#### `POST /agent-query` - Tool invocation
Direct school_events_search tool
```bash
curl -X POST http://localhost:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{"question": "Find art classes for kids"}'
```

#### `POST /multi-agent-query` - Multi-agent system
Routes between LocalEvents and WebSearch agents
```bash
curl -X POST http://localhost:8000/multi-agent-query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest STEM events?"}'
```

### Configuration Endpoints

#### `GET /retrieval-methods` - List methods
```json
{
  "methods": {
    "original": "Original RAG (k=4)",
    "naive": "Naive Retrieval (k=10, LCEL)"
  }
}
```

#### `GET /retrieval-method` - Get active method
```json
{
  "active_method": "naive"
}
```

#### `POST /retrieval-method` - Switch method
```bash
curl -X POST http://localhost:8000/retrieval-method \
  -H "Content-Type: application/json" \
  -d '{"method": "original"}'
```

### Evaluation Endpoints

#### `POST /evaluate-ragas` - Run evaluation
Evaluates current method (takes 1-2 minutes)
```bash
curl -X POST http://localhost:8000/evaluate-ragas
```

**Response:**
```json
{
  "message": "RAGAS evaluation completed",
  "test_questions_count": 10,
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.92,
    "context_precision": 0.78,
    "context_recall": 0.88
  }
}
```

## 🎨 Features

### 1. Event Database (65 Events)
- 🏕️ Camps, 🎯 Challenges, 🎭 Auditions
- ⚽ Sports, 🎨 Arts, 📚 Academic, 🎪 Community

### 2. RAG Pipeline Comparison
- **Original (k=4)**: Faster, focused
- **Naive (k=10)**: Comprehensive, better recall

### 3. Multi-Agent System
- **LocalEvents**: Searches event database
- **WebSearch**: Real-time Tavily search
- **Supervisor**: Intelligent routing

### 4. RAGAS Evaluation
- Faithfulness: Hallucination detection
- Answer Relevancy: Question relevance
- Context Precision: Retrieval quality
- Context Recall: Context completeness

### 5. Frontend Features
- Event browser with 65 events
- RAG-powered chat interface
- Evaluation dashboard
- Method comparison tool

## 🔍 Example Queries

```
"What coding programs are available?"
"Show me art classes for kids"
"Tell me about the robotics club"
"Events for middle school students"
"Affordable coding classes for 10-year-olds"
```

## 📊 Performance

- Query response: 2-5 seconds
- RAGAS evaluation: 60-120 seconds
- Total events: 65
- Vector chunks: ~200+
- Embedding dimension: 1536

## 🐛 Troubleshooting

**Port 8000 in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Import errors:**
```bash
pip install --upgrade -r requirements.txt
```

**Cannot connect (frontend):**
- Verify backend on port 8000
- Check CORS in main.py
- Test: `curl http://localhost:8000/health`

## 📝 Development

### Adding Events
1. Create JSON in `backend/data/`
2. Restart backend
3. Verify: `curl http://localhost:8000/events | grep "Event Name"`

### Modifying Retrieval
```python
# Change k value (main.py ~250)
naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# Change chunk size (main.py ~200)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)
```

### Frontend Customization
```css
/* Theme colors (index.css) */
:root {
  --primary-blue: #1976D2;
  --dark-blue: #0D47A1;
  --light-blue: #90CAF9;
}
```

## 📚 Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [RAGAS Docs](https://docs.ragas.io/)
- [React Docs](https://react.dev/)

## 🤝 Contributing

**Code Style:**
- Python: PEP 8
- JavaScript: ESLint (react-app)

**Testing:**
```bash
# Backend
cd backend
pytest tests/

# Frontend
cd frontend
npm test
```

## 🎓 Certification Project

**Course**: AI Engineering (AIE7)  
**Module**: Production RAG Systems  

**Topics Covered**:
- RAG pipeline implementation
- Multi-agent systems with LangGraph
- RAGAS evaluation framework
- Full-stack development
- Vector databases (Qdrant, ChromaDB)
- LangChain tool creation
- React component architecture

---

## Technologies

- **Framework**: FastAPI + React
- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Stores**: Qdrant (in-memory), ChromaDB
- **Agent Framework**: LangChain, LangGraph
- **Evaluation**: RAGAS
- **Search**: Tavily API

## API Documentation

Interactive Swagger UI: `http://localhost:8000/docs`

## License

Educational project for AI Makerspace certification.

---

**Built with ❤️ for AI Makerspace**

**Last Updated**: October 19, 2025
