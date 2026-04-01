#!/bin/bash
# Development mode — starts FastAPI backend + Vite dev server
# Backend: http://localhost:8000
# Frontend: http://localhost:8080

cd "$(dirname "$0")"
source venv/bin/activate
python run_dashboard.py
