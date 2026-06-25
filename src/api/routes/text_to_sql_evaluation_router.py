"""
text_to_sql_evaluation_router.py — FastAPI router for evaluation endpoints.

Business logic stays in the use case — this file only handles HTTP concerns:
  - Request parsing
  - Dependency injection
  - Error → HTTP status code mapping
  - Response serialisation
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.container import get_run_dataset_use_case
from src.application.dto.run_dataset_request import RunDatasetRequest
from src.application.dto.run_dataset_response import RunDatasetResponse
from src.application.use_cases.run_dataset_evaluation_use_case import (
    RunDatasetEvaluationUseCase,
)
from src.infrastructure.langfuse.langfuse_dataset_provider import DatasetNotFoundError

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/text-to-sql/evaluation",
    tags=["Text-to-SQL Evaluation"],
)


@router.post(
    "/run-single-dataset",
    response_model=RunDatasetResponse,
    status_code=status.HTTP_200_OK,
    summary="Run evaluation against a Langfuse dataset",
    description=(
        "Loads a Langfuse dataset by name, resolves allowed tables "
        "(production tables + additional_tables), runs every item against "
        "the Text-to-SQL agent, executes both expected and generated SQL "
        "through Trino, evaluates results using all registered evaluators, "
        "and stores traces + scores back in Langfuse."
    ),
    responses={
        200: {"description": "Evaluation completed successfully"},
        404: {"description": "Dataset not found in Langfuse"},
        422: {"description": "Invalid request body"},
        503: {"description": "Langfuse or agent unavailable"},
    },
)
async def run_single_dataset(
    request: RunDatasetRequest,
    use_case: RunDatasetEvaluationUseCase = Depends(get_run_dataset_use_case),
) -> RunDatasetResponse:
    """
    Execute a full dataset evaluation run.

    Returns a comprehensive evaluation summary with:
    - Pass/fail counts and rates
    - Latency distribution (p50/p95/p99)
    - Per-evaluator accuracy scores
    - Composite weighted score
    - Failure analysis breakdown
    - Langfuse trace ID for drill-down
    """
    logger.info(
        "router.run_single_dataset",
        dataset_name=request.dataset_name,
        additional_tables=request.additional_tables,
    )

    try:
        return await use_case.execute(request)

    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(
            "router.run_single_dataset.error",
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Evaluation failed: {exc}",
        ) from exc
