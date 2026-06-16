#!/bin/bash

# Backend Development Script
# Runs FastAPI with hot reload

cd "$(dirname "$0")"

echo "🚀 Starting FlowScope AI Backend (Development Mode)"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Create data directory
mkdir -p data

echo ""
echo "✅ Backend ready!"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Run with hot reload
python main.py
