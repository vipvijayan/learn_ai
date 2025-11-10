#!/usr/bin/env python3
"""
Standalone script to run RAGAS evaluation in a separate process.
This avoids uvloop compatibility issues by running in a fresh Python process.
"""

import sys
import json
import asyncio
from app.evaluation.ragas_evaluator import RAGASEvaluator


def main():
    """Run RAGAS evaluation from command line arguments."""
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: run_ragas_standalone.py '<json_data>'"}))
        sys.exit(1)
    
    try:
        # Parse input data
        data = json.loads(sys.argv[1])
        query_data = data["query_data"]
        evaluation_name = data["evaluation_name"]
        
        # Create evaluator and run evaluation
        # Use standard asyncio event loop, not uvloop
        evaluator = RAGASEvaluator()
        result = evaluator.evaluate_responses(
            queries_and_responses=[query_data],
            evaluation_name=evaluation_name
        )
        
        # Output result as JSON
        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        import traceback
        error_result = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
