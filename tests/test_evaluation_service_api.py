import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from src.main import app
from src.application.dto.run_dataset_response import RunDatasetResponse
from src.domain.entities.agent_response import AgentResponse
from src.domain.entities.dataset_item import DatasetItem
from src.domain.entities.evaluation_result import EvaluationResult
from src.domain.entities.query_result import QueryResult

client = TestClient(app)

@pytest.fixture
def mock_agent_client():
    with patch("src.infrastructure.text_to_sql_agent.text_to_sql_agent_client.TextToSqlAgentClient.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = AgentResponse(
            thread_id="test_thread",
            status="completed",
            sql_query="SELECT * FROM minio.test_schema.test_table",
            sql_explanation="Mock explanation",
        )
        yield mock_run

@pytest.fixture
def mock_backend_resolver():
    with patch("src.application.use_cases.run_dataset_evaluation_use_case.BackendTableResolver.get_production_tables", new_callable=AsyncMock) as mock_resolve:
        mock_resolve.return_value = ["minio.test_schema.test_table"]
        yield mock_resolve

@pytest.fixture
def mock_langfuse_dataset():
    with patch("src.infrastructure.langfuse.langfuse_dataset_provider.LangfuseDatasetProvider.get_dataset", new_callable=AsyncMock) as mock_get_ds:
        mock_get_ds.return_value = [
            DatasetItem(
                id="item_1",
                input={
                    "question_id": "q1",
                    "query": "Show me the data",
                    "table_id": "test-table-id",
                    "catalog_name": "minio",
                    "schema_name": "test_schema",
                },
                expected_output={"sql": "SELECT * FROM test_table"},
                metadata={},
                raw={}
            )
        ]
        yield mock_get_ds

@pytest.fixture
def mock_db_evaluator():
    with patch("src.domain.services.evaluation_engine.EvaluationEngine.run_all", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = [
            EvaluationResult(
                evaluator_name="contains_accuracy",
                score=1.0,
                passed=True,
                details={}
            )
        ]
        yield mock_eval

@pytest.fixture
def mock_query_executor():
    with patch("src.infrastructure.trino.trino_query_executor.TrinoQueryExecutor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = QueryResult(
            success=True,
            rows=[[1]],
            columns=["id"],
            row_count=1,
            execution_time_ms=10.0,
            error=None
        )
        yield mock_exec

@pytest.mark.asyncio
async def test_run_single_dataset_endpoint(mock_agent_client, mock_backend_resolver, mock_langfuse_dataset, mock_db_evaluator, mock_query_executor):
    payload = {
        "dataset_name": "text2sql_sandbox_test-table-id",
        "additional_tables": ["minio.test_schema.test_table"]
    }
    
    response = client.post("/text-to-sql/evaluation/run-single-dataset", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert data["total_cases"] == 1
    assert data["accuracy"]["contains_accuracy"] == 1.0
    assert data["failure_rate"] == 0.0
    
    # Verify the mocked agent was called with correct parameters
    mock_agent_client.assert_called_once()
    args, kwargs = mock_agent_client.call_args
    assert kwargs["query"] == "Show me the data"
    assert kwargs["allowed_tables"] == ["minio.test_schema.test_table"]
