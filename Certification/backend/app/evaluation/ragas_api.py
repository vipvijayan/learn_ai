from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.config import config
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

evaluation_buffer = []  # This should be imported or managed globally if needed

class EvaluationRequest(BaseModel):
    """Request model for RAGAS evaluation"""
    clear_buffer: bool = True  # Whether to clear the buffer after evaluation
    evaluation_name: str = "multi_agent_evaluation"


@router.post("/evaluation/run")
async def run_evaluation(request: EvaluationRequest):
    if not config.ENABLE_RAGAS_EVALUATION:
        logger.info("RAGAS evaluation is globally disabled by config.")
        return {
            "status": "disabled",
            "message": "RAGAS evaluation is disabled by global configuration."
        }
    try:
        global evaluation_buffer
        logger.info("="*80)
        logger.info(f"🔬 RAGAS Evaluation: {request.evaluation_name}")
        logger.info("="*80)
        if not evaluation_buffer or len(evaluation_buffer) == 0:
            logger.warning("⚠️ No queries in evaluation buffer")
            return {
                "status": "error",
                "message": "No queries to evaluate. Please run some queries through /multi-agent-query first.",
                "queries_needed": 10,
                "queries_collected": 0
            }
        logger.info(f"� Evaluating {len(evaluation_buffer)} queries from buffer")
        queries_and_responses = []
        for item in evaluation_buffer:
            queries_and_responses.append({
                "user_input": item["user_input"],
                "response": item["response"],
                "retrieved_contexts": item.get("retrieved_contexts", [])
            })
        evaluator = RAGASEvaluator()
        logger.info("📈 Running RAGAS evaluation...")
        results = evaluator.evaluate_responses(
            queries_and_responses=queries_and_responses,
            evaluation_name=request.evaluation_name
        )
        results["queries_evaluated"] = len(evaluation_buffer)
        results["buffer_cleared"] = request.clear_buffer
        if request.clear_buffer:
            logger.info("🧹 Clearing evaluation buffer")
            evaluation_buffer.clear()
        logger.info("="*80)
        logger.info("✅ Evaluation Complete")
        logger.info("="*80)
        return {
            "status": "success",
            "evaluation_name": request.evaluation_name,
            "results": results
        }
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation/buffer")
async def get_evaluation_buffer_status():
    global evaluation_buffer
    return {
        "queries_collected": len(evaluation_buffer),
        "buffer_sample": evaluation_buffer[-5:] if len(evaluation_buffer) > 0 else [],
        "message": f"Collected {len(evaluation_buffer)} queries. Need at least 1 query to run evaluation."
    }

@router.post("/evaluation/buffer/clear")
async def clear_evaluation_buffer():
    global evaluation_buffer
    count = len(evaluation_buffer)
    evaluation_buffer.clear()
    logger.info(f"🧹 Cleared evaluation buffer ({count} queries removed)")
    return {
        "status": "success",
        "message": f"Cleared {count} queries from evaluation buffer"
    }
