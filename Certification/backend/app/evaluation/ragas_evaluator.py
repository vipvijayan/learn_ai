"""
RAGAS Evaluation Module for School Events RAG System

Simple module to evaluate actual multi-agent responses using RAGAS metrics.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from ragas import EvaluationDataset, evaluate, RunConfig, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import Faithfulness, ResponseRelevancy

# Configure logging
logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Simple evaluator for School Events RAG system using RAGAS.
    Only evaluates provided responses - no data generation or complex workflows.
    """
    
    def __init__(self, results_path: str = "generated_results/evaluations/"):
        """Initialize the RAGAS evaluator."""
        self.results_path = results_path
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM for evaluation
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        logger.info(f"✅ RAGASEvaluator initialized")
    
    def evaluate_responses(
        self,
        queries_and_responses: List[Dict[str, Any]],
        evaluation_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """
        Evaluate responses using RAGAS metrics.
        
        Args:
            queries_and_responses: List of dicts with:
                - user_input: The user query
                - response: The agent response
                - retrieved_contexts: List of context strings
            evaluation_name: Name for this evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"📊 Evaluating {len(queries_and_responses)} responses")
        
        try:
            # Create RAGAS samples
            samples = []
            for item in queries_and_responses:
                sample = SingleTurnSample(
                    user_input=item["user_input"],
                    response=item["response"],
                    retrieved_contexts=item.get("retrieved_contexts", [])
                )
                samples.append(sample)
            
            # Create evaluation dataset
            evaluation_dataset = EvaluationDataset(samples=samples)
            
            # Define metrics
            metrics = [
                Faithfulness(),
                ResponseRelevancy()
            ]
            
            logger.info(f"📈 Running evaluation with Faithfulness and ResponseRelevancy...")
            
            # Run evaluation
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                run_config=RunConfig(timeout=360, max_workers=4)
            )
            
            logger.info("✅ Evaluation complete")
            
            # Format results
            # EvaluationResult object has attributes, not dict keys
            results_dict = {
                "evaluation_name": evaluation_name,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(samples),
                "metrics": {
                    "faithfulness": float(result.to_pandas()["faithfulness"].mean() if "faithfulness" in result.to_pandas().columns else 0),
                    "response_relevancy": float(result.to_pandas()["answer_relevancy"].mean() if "answer_relevancy" in result.to_pandas().columns else 0)
                },
                "samples": [
                    {
                        "query": item["user_input"],
                        "response_preview": item["response"][:200] + "..." if len(item["response"]) > 200 else item["response"]
                    }
                    for item in queries_and_responses
                ]
            }
            
            # Save results
            self._save_results(results_dict)
            
            logger.info(f"📊 Results:")
            logger.info(f"   Faithfulness: {results_dict['metrics']['faithfulness']:.4f}")
            logger.info(f"   Response Relevancy: {results_dict['metrics']['response_relevancy']:.4f}")
            
            return results_dict
            
        except Exception as e:
            logger.error(f"❌ Evaluation error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{results['evaluation_name']}_{timestamp}.json"
        filepath = os.path.join(self.results_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Saved to: {filepath}")
        return filepath

