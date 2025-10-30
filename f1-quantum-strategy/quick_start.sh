#!/bin/bash
# Quick fix script for Python 3.13 compatibility

echo "🔧 F1 Quantum Strategy - Quick Fix for Python 3.13"
echo "==================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    echo "Please run this script from the f1-quantum-strategy folder"
    exit 1
fi

echo "✅ Found project files"
echo "✅ Python version: $(python3 --version)"
echo ""

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "🔧 Creating new virtual environment..."
python3 -m venv venv

# Activate it
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies one by one with error handling
echo ""
echo "📦 Installing dependencies (this will take 2-3 minutes)..."
echo ""

# Install in order of dependency
echo "   [1/8] Installing FastAPI..."
pip install "fastapi>=0.115.0" --quiet

echo "   [2/8] Installing Uvicorn..."
pip install "uvicorn[standard]>=0.32.0" --quiet

echo "   [3/8] Installing Pydantic..."
pip install "pydantic>=2.10.0" --quiet

echo "   [4/8] Installing NumPy (this may take a minute)..."
pip install "numpy>=1.26.0,<2.3.0" --quiet

echo "   [5/8] Installing Pandas (this may take a minute)..."
pip install "pandas>=2.2.0" --quiet

echo "   [6/8] Installing Qiskit..."
pip install "qiskit>=1.3.0" --quiet

echo "   [7/8] Installing Qiskit-Aer..."
pip install "qiskit-aer>=0.15.0" --quiet

echo "   [8/8] Installing utilities..."
pip install requests python-multipart --quiet

echo ""
echo "=============================================="
echo "✅ Installation Complete!"
echo "=============================================="
echo ""
echo "🧪 Running quick test..."
echo ""

# Quick test
python3 -c "
import fastapi
import qiskit
import numpy as np
import pandas as pd
print('✅ FastAPI version:', fastapi.__version__)
print('✅ Qiskit version:', qiskit.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ Pandas version:', pd.__version__)
print('')
print('🎉 All dependencies loaded successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "🚀 Ready to start!"
    echo "=============================================="
    echo ""
    echo "Run the backend with:"
    echo "   python3 main.py"
    echo ""
    echo "Or test with:"
    echo "   python3 quick_test.py"
    echo ""
else
    echo ""
    echo "❌ There was an issue with the installation"
    echo "Try running: pip install -r requirements.txt"
    echo ""
fi