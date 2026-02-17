#!/bin/bash
#
# System Call Optimization Platform - Run Script
#

echo "=================================================="
echo "   SysCall AI - System Call Optimization Platform"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run the application
echo ""
echo "Starting server..."
echo "Open http://localhost:5000 in your browser"
echo ""

cd backend
python app.py
