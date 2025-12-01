#!/bin/bash
# Start MLflow Tracking Server

set -e

echo "🚀 Starting MLflow Tracking Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000

echo "✅ MLflow UI available at http://localhost:5000"
