#!/bin/bash

# RAG Application Startup Script
echo "🚀 Starting PDF & Excel RAG Application..."

# Check if we're in the right directory
if [ ! -d "api" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the 03_End-to-End_RAG directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo ""
echo "📋 Setting up the application..."

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd api
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found in api directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

cd ..

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend

if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found in frontend directory"
    exit 1
fi

npm install

cd ..

echo ""
echo "✅ Dependencies installed successfully!"
echo ""
echo "🌟 Starting the application..."
echo ""

# Start the API server in the background
echo "🔧 Starting API server (http://localhost:8000)..."
cd api
source venv/bin/activate
nohup python main.py > ../api.log 2>&1 &
API_PID=$!
cd ..

# Wait a moment for the API to start
sleep 3

# Start the frontend
echo "🎨 Starting frontend (http://localhost:3000)..."
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "🎉 Application started successfully!"
echo ""
echo "📍 Access points:"
echo "   • Frontend: http://localhost:3000"
echo "   • API: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "   • API logs: ../api.log"
echo ""
echo "⏹️  To stop the application:"
echo "   • Press Ctrl+C to stop the frontend"
echo "   • Run: kill $API_PID (to stop the API)"
echo ""
echo "🔧 Troubleshooting:"
echo "   • Make sure ports 3000 and 8000 are available"
echo "   • Check api.log for backend errors"
echo "   • Ensure all dependencies are installed"

# Wait for frontend to start
wait $FRONTEND_PID