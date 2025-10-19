#!/bin/bash
# Quick activation script for Certification virtual environment
source "$(dirname "$0")/venv/bin/activate"
echo "✓ Virtual environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
