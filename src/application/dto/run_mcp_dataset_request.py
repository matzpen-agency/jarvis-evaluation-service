"""
run_mcp_dataset_request.py — API request DTO for MCP-agent dataset evaluation.

Used by POST /text-to-sql/evaluation/run-mcp-dataset.
Unlike RunDatasetRequest (which resolves allowed_tables from the backend),
this request passes a connection_id directly to the new agent — the agent
itself uses that to resolve the appropriate schema/tables via jeen-metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunMcpDatasetRequest(BaseModel):
    """
    Request body for POST /text-to-sql/evaluation/run-mcp-dataset.
    """

    dataset_name: str = Field(
        ...,
        description="Langfuse dataset name to evaluate.",
        examples=["sales_questions"],
    )
    connection_id: int = Field(
        ...,
        description=(
            "Numeric connection ID passed to the MCP agent's chat_with_agent tool. "
            "This maps to JEEN_METADATA_CONNECTION_ID in the agent config and determines "
            "which database connection / schema the agent queries."
        ),
        examples=[89],
    )
    question_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific item IDs to evaluate.",
        examples=[["q-1", "q-2"]],
    )
    limit: int | None = Field(
        default=None,
        ge=0,
        description="Optional limit on the number of cases to evaluate.",
        examples=[5],
    )
    offset: int | None = Field(
        default=None,
        ge=0,
        description="Optional offset of cases to evaluate.",
        examples=[0],
    )
