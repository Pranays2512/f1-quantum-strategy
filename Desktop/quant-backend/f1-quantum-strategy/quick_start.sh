#!/bin/bash
# F1 Quantum Strategy Backend - Quick Start Script
# This script automates the entire setup process

echo "🏎️  F1 Quantum Strategy Backend - Automated Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.9 or higher from python.org"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed!"
    echo "Please install pip: python3 -m ensurepip --upgrade"
    exit 1
fi

echo "✅ pip found"
echo ""

# Create project directory
echo "📁 Creating project directory..."
mkdir -p f1-quantum-strategy
cd f1-quantum-strategy

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Mac/Linux
    source venv/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install --quiet fastapi uvicorn qiskit qiskit-aer numpy pandas pydantic requests

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully!"
else
    echo "❌ Installation failed. Please check your internet connection."
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "📋 Next Steps:"
echo "1. Copy your Python files (main.py, quantum_strategy_engine.py, etc.) into:"
echo "   $(pwd)"
echo ""
echo "2. Start the backend:"
echo "   python main.py"
echo ""
echo "3. Test at: http://localhost:8000/docs"
echo ""
echo "💡 To activate virtual environment later:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   venv\\Scripts\\activate"
else
    echo "   source venv/bin/activate"
fi
echo ""