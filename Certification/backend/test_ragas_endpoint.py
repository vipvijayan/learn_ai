#!/usr/bin/env python3
"""
Test script for RAGAS evaluation endpoint
"""
import requests
import json
import time

print("🧪 Testing RAGAS Evaluation Endpoint...")
print("=" * 80)

# Check if server is running
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    print("✅ Backend server is running")
except requests.exceptions.RequestException:
    print("❌ Backend server is NOT running")
    print("   Start it with: cd backend && . venv/bin/activate && uvicorn main:app --reload --port 8000")
    exit(1)

# Test RAGAS endpoint
print("\n🔬 Running RAGAS evaluation...")
print("   This may take 30-60 seconds...\n")

try:
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/evaluate-ragas",
        headers={"Content-Type": "application/json"},
        timeout=120  # 2 minute timeout
    )
    duration = time.time() - start_time
    
    print(f"⏱️  Completed in {duration:.1f} seconds")
    print("=" * 80)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ RAGAS Evaluation Successful!")
        print("\n📊 Metrics:")
        for metric, value in result.get("metrics", {}).items():
            print(f"   {metric:.<30} {value:.4f}")
        print(f"\n📝 Test Questions: {result.get('test_questions_count')}")
        print(f"💾 Files Generated:")
        for file_path in result.get("files_generated", []):
            print(f"   - {file_path}")
    else:
        print(f"❌ Error ({response.status_code}):")
        print(f"   {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out (>120s)")
    print("   RAGAS evaluation is taking longer than expected")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("=" * 80)
