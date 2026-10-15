"""
run_mcp_dataset_evaluation_use_case.py — Use case for MCP-agent dataset evaluation.

Unlike RunDatasetEvaluationUseCase (which resolves allowed_tables from the backend),
this use case skips table resolution entirely. The new agent receives a connection_id
and resolves the schema itself via jeen-metadata. We pass an empty allowed_tables to
the runner (the MCP client ignores it) and forward the connection_id via the agent
client's kwargs.
"""

from __future__ import annotations

import structlog

from src.application.dto.run_dataset_response import RunDatasetResponse
from src.application.dto.run_mcp_dataset_request import RunMcpDatasetRequest
from src.application.use_cases.run_dataset_evaluation_use_case import (
    RunDatasetEvaluationUseCase,
)
from src.domain.services.evaluation_runner import EvaluationRunner

logger = structlog.get_logger(__name__)


class RunMcpDatasetEvaluationUseCase:
    """
    Application-layer use case for evaluating datasets via the new MCP agent.

    Key difference from RunDatasetEvaluationUseCase:
      - No backend table resolution (connection_id replaces allowed_tables)
      - connection_id is forwarded to the MCP client via EvaluationRunner.run()
    """

    def __init__(self, runner: EvaluationRunner) -> None:
        self._runner = runner

    async def execute(self, request: RunMcpDatasetRequest) -> RunDatasetResponse:
        """
        Run a dataset evaluation against the MCP agent.

        Raises:
            DatasetNotFoundError: If the Langfuse dataset doesn't exist.
        """
        logger.info(
            "use_case.run_mcp_dataset_evaluation.start",
            dataset_name=request.dataset_name,
            connection_id=request.connection_id,
        )

        dataset_run = await self._runner.run(
            dataset_name=request.dataset_name,
            # MCP agent ignores allowed_tables — connection_id takes precedence.
            # We pass an empty list so the evaluation pipeline doesn't break if
            # any evaluator checks allowed_tables defensively.
            allowed_tables=[],
            question_ids=request.question_ids,
            limit=request.limit,
            offset=request.offset,
            # Forwarded through EvaluationRunner → agent_client.run(**kwargs)
            agent_kwargs={"connection_id": request.connection_id},
        )

        return RunDatasetEvaluationUseCase._to_response(dataset_run)
