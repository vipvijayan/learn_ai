import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _prepare_contexts(final_response: str, result: Dict[str, Any]) -> List[str]:
    """Aggregate contexts from the final response and intermediate messages."""
    contexts = [final_response]

    try:
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            logger.info(f"📝 Found {len(messages)} messages in result")
            for msg in messages:
                content = getattr(msg, "content", None)
                if content and content != final_response:
                    contexts.append(str(content)[:1000])  # limit context size
    except Exception as ctx_error:  # pragma: no cover - defensive
        logger.warning(f"⚠️ Could not extract additional contexts: {ctx_error}")

    logger.info(f"📚 Using {len(contexts)} context(s) for evaluation")
    return contexts


def _build_query_data(
    question: str,
    final_response: str,
    contexts: List[str],
    final_agent_name: Optional[str],
) -> Dict[str, Any]:
    """Create the payload consumed by the standalone RAGAS runner."""
    query_data = {
        "user_input": question,
        "response": final_response,
        "retrieved_contexts": contexts,
        "agent_used": final_agent_name,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        "📊 Query data prepared: question=%s..., response length=%d, contexts=%d",
        question[:50],
        len(final_response),
        len(contexts),
    )
    return query_data


def _run_ragas_subprocess(eval_input: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the standalone RAGAS runner via subprocess."""
    script_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "run_ragas_standalone.py",
    )
    script_path = os.path.abspath(script_path)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Standalone evaluator not found at {script_path}")
    python_path = sys.executable

    env = os.environ.copy()
    env["GIT_PYTHON_REFRESH"] = "quiet"

    result = subprocess.run(
        [python_path, script_path, json.dumps(eval_input)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"RAGAS subprocess failed: {result.stderr}")

    return json.loads(result.stdout)


async def run_websocket_ragas_evaluation(
    *,
    question: str,
    final_response: str,
    result: Dict[str, Any],
    final_agent_name: Optional[str],
    websocket,
) -> Dict[str, Any]:
    """Execute RAGAS evaluation for WebSocket flow and emit progress."""
    logger.info("🔬 Starting automatic RAGAS evaluation for WebSocket...")

    contexts = _prepare_contexts(final_response, result)
    query_data = _build_query_data(question, final_response, contexts, final_agent_name)

    eval_input = {
        "query_data": query_data,
        "evaluation_name": f"auto_eval_ws_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    logger.info("🔧 Running RAGAS in separate process to avoid uvloop conflicts...")
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        evaluation_result = await loop.run_in_executor(
            executor,
            _run_ragas_subprocess,
            eval_input,
        )

    evaluation_payload = {
        "type": "evaluation",
        "evaluation": {
            "faithfulness": evaluation_result["metrics"]["faithfulness"],
            "response_relevancy": evaluation_result["metrics"]["response_relevancy"],
            "status": "completed",
        },
    }

    await websocket.send_json(evaluation_payload)
    logger.info(
        "✅ Evaluation sent: Faithfulness=%.3f, Relevancy=%.3f",
        evaluation_result["metrics"]["faithfulness"],
        evaluation_result["metrics"]["response_relevancy"],
    )

    return evaluation_result
