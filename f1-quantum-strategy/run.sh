#!/bin/bash
# =====================================================
# 🏎️  F1 Quantum Strategy Backend - Run Script
# =====================================================

# Exit immediately if any command fails
set -e

# Step 1: Activate virtual environment
echo "🔹 Activating virtual environment..."
source venv/bin/activate

# Step 2: Install dependencies
echo "🔹 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 3: Run the backend server
echo "🚀 Starting F1 Quantum Strategy Backend..."
python main.py
