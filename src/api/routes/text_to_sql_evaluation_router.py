"""
text_to_sql_evaluation_router.py — FastAPI router for evaluation endpoints.

Business logic stays in the use case — this file only handles HTTP concerns:
  - Request parsing
  - Dependency injection
  - Error → HTTP status code mapping
  - Response serialisation
"""

from __future__ import annotations

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.api.dependencies.container import (
    get_run_dataset_use_case,
    get_run_mcp_dataset_use_case,
)
from src.application.dto.run_dataset_request import RunDatasetRequest
from src.application.dto.run_dataset_response import (
    RunDatasetCaseResultDTO,
    RunDatasetResponse,
)
from src.application.dto.run_mcp_dataset_request import RunMcpDatasetRequest
from src.application.jobs.evaluation_job_store import (
    EvaluationJob,
    EvaluationJobStore,
    get_job_store,
)
from src.application.use_cases.run_dataset_evaluation_use_case import (
    RunDatasetEvaluationUseCase,
)
from src.application.use_cases.run_mcp_dataset_evaluation_use_case import (
    RunMcpDatasetEvaluationUseCase,
)
from src.ports.dataset_provider import DatasetNotFoundError

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


@router.post(
    "/run-new-evaluation",
    response_model=RunDatasetResponse,
    status_code=status.HTTP_200_OK,
    summary="Run evaluation against a Langfuse dataset using the MCP agent",
    description=(
        "Loads a Langfuse dataset by name, runs every item against the updated "
        "Text-to-SQL MCP agent (chat_with_agent tool), passing connection_id instead "
        "of allowed_tables — the agent resolves schema via jeen-metadata internally. "
        "Executes expected SQL through Trino, evaluates results using all registered "
        "evaluators, and stores traces + scores back in Langfuse."
    ),
    responses={
        200: {"description": "Evaluation completed successfully"},
        404: {"description": "Dataset not found in Langfuse"},
        422: {"description": "Invalid request body"},
        503: {"description": "Langfuse or MCP agent unavailable"},
    },
)
async def run_new_evaluation(
    request: RunMcpDatasetRequest,
    use_case: RunMcpDatasetEvaluationUseCase = Depends(get_run_mcp_dataset_use_case),
) -> RunDatasetResponse:
    """
    Execute a full dataset evaluation run against the updated MCP-based agent.

    Unlike run-single-dataset, this endpoint does NOT resolve allowed_tables
    from the backend. Instead it passes connection_id directly to the agent,
    which resolves the appropriate schema itself via jeen-metadata.

    Returns the same comprehensive evaluation summary as run-single-dataset.
    """
    logger.info(
        "router.run_new_evaluation",
        dataset_name=request.dataset_name,
        connection_id=request.connection_id,
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
            "router.run_new_evaluation.error",
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation failed.",
        ) from exc


# ── Background jobs ────────────────────────────────────────────────────────────


class EvaluationJobResponse(BaseModel):
    """Status of a background evaluation job."""

    job_id: str
    status: str  # running | completed | failed
    completed_items: int = 0
    total_items: int | None = None
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    # Questions finished so far, with their scores (filled while running)
    completed_cases: list[RunDatasetCaseResultDTO] = []
    result: RunDatasetResponse | None = None


def _job_response(job: EvaluationJob) -> EvaluationJobResponse:
    return EvaluationJobResponse(
        job_id=job.job_id,
        status=job.status,
        completed_items=job.completed_items,
        total_items=job.total_items,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
        completed_cases=list(job.completed_cases),
        result=job.result,
    )


@router.post(
    "/run-new-evaluation/jobs",
    response_model=EvaluationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a run-new-evaluation in the background",
    description=(
        "Same evaluation as POST /run-new-evaluation, but returns a job id "
        "immediately. Poll GET /jobs/{job_id} until status is completed or "
        "failed; the completed job carries the full RunDatasetResponse."
    ),
)
async def submit_new_evaluation_job(
    request: RunMcpDatasetRequest,
    use_case: RunMcpDatasetEvaluationUseCase = Depends(get_run_mcp_dataset_use_case),
    job_store: EvaluationJobStore = Depends(get_job_store),
) -> EvaluationJobResponse:
    logger.info(
        "router.submit_new_evaluation_job",
        dataset_name=request.dataset_name,
        connection_id=request.connection_id,
    )

    async def run(progress_callback) -> RunDatasetResponse:
        return await use_case.execute(
            request,
            progress_callback=progress_callback,
            case_callback=getattr(progress_callback, "add_case", None),
        )

    job = job_store.submit(
        run,
        metadata={
            "dataset_name": request.dataset_name,
            "connection_id": request.connection_id,
        },
    )
    return _job_response(job)


@router.get(
    "/jobs/{job_id}",
    response_model=EvaluationJobResponse,
    summary="Get the status (and result) of a background evaluation job",
    responses={404: {"description": "Unknown job (or the service restarted)"}},
)
async def get_evaluation_job(
    job_id: str,
    job_store: EvaluationJobStore = Depends(get_job_store),
) -> EvaluationJobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation job '{job_id}' not found",
        )
    return _job_response(job)
