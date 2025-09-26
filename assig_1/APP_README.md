# AI Chat Application

This is a full-stack AI chat application built with React (TypeScript) frontend and FastAPI backend, deployed on Vercel.

## 🚀 Live Deployment

**Production URL**: https://assig-1-kcfk5ngnl-vipin-vijayan-nairs-projects.vercel.app

**Vercel Dashboard**: https://vercel.com/vipin-vijayan-nairs-projects/assig-1/settings

## Architecture

- **Frontend**: React TypeScript app
- **Backend**: FastAPI Python server (serverless on Vercel)
- **AI Integration**: OpenAI API for chat completions
- **Deployment**: Vercel with automatic builds and deployments

## Setup and Running Locally

### Prerequisites
- Node.js and npm
- Python 3.13
- OpenAI API key

### Installation

1. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

2. **Backend Setup**:
   ```bash
   cd api
   python3 -m venv venv
   # Note: Virtual environment packages are already installed
   ```

### Running the Application Locally

1. **Start the Backend API** (Terminal 1):
   ```bash
   cd api
   python3 start_server.py
   ```
   The API will be available at: http://localhost:8000

2. **Start the Frontend** (Terminal 2):
   ```bash
   cd frontend
   npm start
   ```
   The web app will be available at: http://localhost:3000

## Deployment

### Vercel Deployment

The application is configured for automatic deployment on Vercel:

1. **Deploy to production**:
   ```bash
   vercel --prod
   ```

2. **Deploy to preview**:
   ```bash
   vercel
   ```

### Vercel Configuration

- **vercel.json**: Configures builds for both frontend and backend
- **API Routes**: Backend FastAPI endpoints are served as serverless functions
- **Static Assets**: Frontend build is served as static files
- **Routing**: API requests are routed to `/api/*` endpoints

## Usage

1. Visit the live deployment URL or run locally
2. Enter your OpenAI API key in the input field
3. Type your message in the textarea
4. Click "Send" to get a response from the AI

## API Endpoints

- `POST /api/chat` - Send a chat message to OpenAI
- `GET /api/health` - Health check endpoint

## Deployment Status

✅ **Frontend Build**: Completed successfully  
✅ **Backend API**: Deployed as serverless functions  
✅ **Production Deployment**: Live on Vercel  
✅ **CORS Configuration**: Properly configured for cross-origin requests  
✅ **Routing**: API and frontend routing working correctly  

## Project Structure

```
assig_1/
├── api/
│   ├── app.py              # FastAPI backend
│   ├── start_server.py     # Local development server
│   ├── requirements.txt    # Python dependencies
│   └── vercel.json         # API-specific Vercel config
├── frontend/
│   ├── src/
│   │   └── App.tsx         # React main component
│   ├── package.json        # Frontend dependencies
│   └── build/              # Production build
├── vercel.json             # Main Vercel deployment config
└── APP_README.md           # This documentation
```

The application is now fully deployed and operational on Vercel! 🎉