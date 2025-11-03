# School Events RAG Application

A simple chat application that helps parents find information about school events and programs using RAG (Retrieval Augmented Generation).

## 🆕 New: Gmail MCP Integration

The application now includes **Gmail MCP (Model Context Protocol) integration** that searches your Gmail inbox for Round Rock school data!

# School Events RAG Application

A simple chat application that helps parents find information about school events and programs using RAG (Retrieval Augmented Generation).

## 🆕 New: Gmail Integration

The application now includes **Gmail integration** that searches your Gmail inbox for Round Rock school data using Google's Gmail API!

### 🔐 Gmail Authentication Required

To use Gmail search, you must authenticate first:

```bash
cd backend
python authenticate_gmail.py
```

👉 **[Complete Gmail Integration Guide →](GMAIL_INTEGRATION_GUIDE.md)** - Comprehensive setup, authentication, and usage guide

## Project Structure

### Documentation
- **[Gmail Authentication](GMAIL_AUTHENTICATION.md)** - How to authenticate for first-time users
- **[Quick Start Guide](README_GMAIL_MCP.md)** - Overview and commands
- **[Setup Checklist](GMAIL_MCP_CHECKLIST.md)** - Step-by-step setup
- **[Detailed Setup](GMAIL_MCP_SETUP.md)** - Complete instructions
- **[Implementation Details](GMAIL_MCP_IMPLEMENTATION.md)** - Technical documentation

## Project Structure

```
Certification/
├── backend/           # FastAPI Python backend
│   ├── main.py       # Main FastAPI application
│   ├── requirements.txt
│   └── start.sh      # Backend startup script
├── frontend/         # React frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── start.sh      # Frontend startup script
└── data/            # School events JSON data
```

## Problem Statement

Parents receive numerous emails from schools about different events and programs, making it hard to remember and find specific information when needed.

## Solution

An AI-powered chat interface that allows parents to ask natural language questions about school events and get relevant information from all available school data.

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Ollama (optional, will fallback to OpenAI if not available)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (if using OpenAI as fallback):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

4. Start the backend:
   ```bash
   python main.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The application will be available at `http://localhost:3000`

## Features

- 💬 Natural language chat interface
- 🔍 RAG-powered search through school events data
- 📱 Responsive design
- 🎯 Context-aware responses with source information
- 🔄 Real-time query processing

## Usage Examples

Ask questions like:
- "What coding programs are available?"
- "Tell me about holiday day camps"
- "What activities are there for middle school students?"
- "When is registration open for sports programs?"

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangChain**: RAG pipeline orchestration
- **ChromaDB**: Vector database for embeddings
- **Ollama/OpenAI**: LLM for response generation

### Frontend
- **React**: User interface framework
- **Axios**: HTTP client for API communication
- **CSS3**: Styling and responsive design

## Data Sources

The application processes JSON files containing school event information from the `../../data/` directory, including:
- Program details and descriptions
- Target age groups and grades
- Registration information
- Schedules and dates
- Organization details