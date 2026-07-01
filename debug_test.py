import asyncio
from tests.conftest import sample_context, sample_dataset_item
from src.domain.evaluators.execution_accuracy_evaluator import ExecutionAccuracyEvaluator

async def main():
    ctx = sample_context(sample_dataset_item())
    evaluator = ExecutionAccuracyEvaluator()
    result = await evaluator.evaluate(ctx)
    print("Score:", result.score)
    print("Details:", result.details)
    
    expected = ctx.expected_result.as_normalised_row_tuples(6)
    generated = ctx.generated_result.as_normalised_row_tuples(6)
    print("Expected:", expected)
    print("Generated:", generated)
    from collections import Counter
    print("Counter eq:", Counter(expected) == Counter(generated))

asyncio.run(main())
