#!/usr/bin/env python3
"""
Startup script for the TeleHealthCareAI FastAPI application.
This script sets up the proper Python path and environment variables.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Set environment variables for OpenRouter
os.environ.setdefault('OPENROUTER_API_KEY', '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')
os.environ.setdefault('OPENROUTER_MODEL', 'mistralai/mistral-small-3.2-24b-instruct:free')
os.environ.setdefault('SITE_URL', 'https://telehealthcareai.com')
os.environ.setdefault('SITE_NAME', 'TeleHealthCareAI')

# Import and run the FastAPI application
from research.app.main import app

if __name__ == "__main__":
    import uvicorn

    
    uvicorn.run(
        "research.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 