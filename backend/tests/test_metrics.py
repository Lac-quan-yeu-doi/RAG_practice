from rag_project.evaluation import evaluate_retrieval, exact_match, token_f1


def test_retrieval_metrics():
    metrics = evaluate_retrieval(
        predictions=[["a", "b"], ["x", "c"]],
        ground_truth=[["a"], ["c"]],
    )
    assert metrics.hit_rate == 1.0
    assert metrics.recall == 1.0
    assert metrics.mean_reciprocal_rank == 0.75


def test_answer_metrics():
    assert exact_match("A Task runs a coroutine.", "a task runs a coroutine") == 1.0
    assert 0.0 < token_f1("run tasks concurrently", "tasks run concurrently") <= 1.0
