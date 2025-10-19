#!/bin/bash

echo "Starting School Events React Frontend..."

# Check if node_modules exists, install if not
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server
echo "Starting React development server..."
npm start