from vforge.training.reporting import compute_throughput_metrics


def test_compute_throughput_scales_counts_by_dp_and_rates_by_world():
    metrics = compute_throughput_metrics(
        local_batch=1,
        local_tokens=867_600,
        dp_size=2,
        world_size=16,
        timing={"step_compute_s": 200.0, "step_s": 220.0},
    )

    assert metrics["global_batch"] == 2.0
    assert metrics["global_tokens"] == 1_735_200.0
    assert metrics["tokens_per_sec"] == 1_735_200.0 / 200.0
    assert metrics["samples_per_sec"] == 2.0 / 200.0
    # Counts scale with DP; per-GPU rate divides by all participating GPUs.
    assert metrics["tokens_per_sec_per_gpu"] == metrics["tokens_per_sec"] / 16.0
    assert metrics["tokens_per_sec_e2e"] == 1_735_200.0 / 220.0


def test_compute_throughput_without_timing_keeps_counts():
    metrics = compute_throughput_metrics(
        local_batch=1,
        local_tokens=1000,
        dp_size=1,
        world_size=1,
        timing=None,
    )

    assert metrics["global_tokens"] == 1000.0
    assert "tokens_per_sec" not in metrics
