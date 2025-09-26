#!/usr/bin/env python3
import sys
import os

# Add the virtual environment's site-packages to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'venv', 'lib', 'python3.13', 'site-packages'))

# Now import uvicorn and run the app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)