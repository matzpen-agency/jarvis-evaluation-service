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
from src.domain.services.evaluation_runner import EvaluationRunner, ProgressCallback
from src.infrastructure.bird.bird_dataset_loader import (
    BIRD_DATASET_PREFIX,
    bird_db_id_for_connection,
)
from src.infrastructure.langfuse.langfuse_dataset_provider import DatasetNotFoundError
from src.ports.connection_metadata_provider import ConnectionMetadataProvider

logger = structlog.get_logger(__name__)


class RunMcpDatasetEvaluationUseCase:
    """
    Application-layer use case for evaluating datasets via the new MCP agent.

    Key difference from RunDatasetEvaluationUseCase:
      - No backend table resolution (connection_id replaces allowed_tables)
      - connection_id is forwarded to the MCP client via EvaluationRunner.run()
    """

    def __init__(
        self,
        runner: EvaluationRunner,
        metadata_provider: ConnectionMetadataProvider | None = None,
    ) -> None:
        self._runner = runner
        self._metadata_provider = metadata_provider

    async def _resolve_dataset_name(self, request: RunMcpDatasetRequest) -> str:
        """
        "bird" means: the BIRD questions of the database this connection holds,
        found by the connection's name (e.g. "movie_platform" → "bird-movie_platform").
        """
        if request.dataset_name.lower() != "bird":
            return request.dataset_name
        name = None
        if self._metadata_provider is not None:
            name = await self._metadata_provider.get_connection_name(request.connection_id)
        db_id = bird_db_id_for_connection(name)
        if not db_id:
            raise DatasetNotFoundError(
                f"Can't tell which BIRD database connection {request.connection_id} holds "
                f"(connection name: {name!r}). Name the connection after the BIRD db_id."
            )
        return f"{BIRD_DATASET_PREFIX}{db_id}"

    async def execute(
        self,
        request: RunMcpDatasetRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> RunDatasetResponse:
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

        dataset_name = await self._resolve_dataset_name(request)
        dataset_run = await self._runner.run(
            dataset_name=dataset_name,
            # MCP agent ignores allowed_tables — connection_id takes precedence.
            # We pass an empty list so the evaluation pipeline doesn't break if
            # any evaluator checks allowed_tables defensively.
            allowed_tables=[],
            question_ids=request.question_ids,
            limit=request.limit,
            offset=request.offset,
            # Forwarded through EvaluationRunner → agent_client.run(**kwargs)
            agent_kwargs={"connection_id": request.connection_id},
            progress_callback=progress_callback,
        )

        return RunDatasetEvaluationUseCase._to_response(dataset_run)
