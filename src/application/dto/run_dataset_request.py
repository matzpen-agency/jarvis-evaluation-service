"""
run_dataset_request.py — API request DTO for dataset evaluation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunDatasetRequest(BaseModel):
    """
    Request body for POST /text-to-sql/evaluation/run-single-dataset.
    """

    dataset_name: str = Field(
        ...,
        description="Langfuse dataset name to evaluate.",
        examples=["sales_questions"],
    )
    additional_tables: list[str] = Field(
        default_factory=list,
        description=(
            "Extra table names to include alongside production-status tables. "
            "These are unioned with the automatically resolved production table list."
        ),
        examples=[["orders", "customers"]],
    )
    question_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific item IDs to evaluate.",
        examples=[["q-1", "q-2"]],
    )
    limit: int | None = Field(
        default=None,
        description="Optional limit on the number of cases to evaluate.",
        examples=[5],
    )
    offset: int | None = Field(
        default=None,
        description="Optional offset of cases to evaluate.",
        examples=[0],
    )
