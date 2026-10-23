from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _gather_gradient_worker(rank, world_size, init_file):
    import os

    from vforge.distributed.ulysses import gather_sequence

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        for mode in (None, "all_reduce", "reduce_scatter"):
            # Explicit policies must take precedence over an invalid environment.
            os.environ["VFORGE_ULYSSES_GATHER_BACKWARD"] = "invalid" if mode else "all_reduce"
            for scale in (False, True):
                local = torch.full((1, 2, 3), rank + 1.0, requires_grad=True)
                output = gather_sequence(
                    local, dim=1, group=dist.group.WORLD, scale_grad=scale, backward_mode=mode
                )
                expected = torch.cat(
                    [torch.full_like(local, peer + 1.0) for peer in range(world_size)], dim=1
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                ((rank + 1) * output.square().sum()).backward()
                factor = sum(range(1, world_size + 1)) * (world_size if scale else 1)
                torch.testing.assert_close(local.grad, 2 * local.detach() * factor, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_gather_backward_policy_preserves_sum_and_scaling(tmp_path):
    mp.spawn(_gather_gradient_worker, args=(2, str(tmp_path / "gather-pg")), nprocs=2, join=True)


def _slice_gradient_worker(rank, world_size, init_file):
    from vforge.distributed.ulysses import pad_to_multiple, slice_sequence

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        for gather_grad in (True, False):
            for scale_grad in (False, True):
                full = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3).requires_grad_()
                reference = full.detach().clone().requires_grad_()
                padded, _ = pad_to_multiple(full, world_size, dim=-2)
                options = {} if gather_grad else {"gather_grad": False}
                local = slice_sequence(
                    padded, dim=-2, group=dist.group.WORLD, scale_grad=scale_grad, **options
                )
                # Reference uses ordinary local slices, with one loss per contributing rank.
                chunks = F.pad(reference, (0, 0, 0, 1)).chunk(world_size, dim=1)
                assert torch.equal(local, chunks[rank])
                ranks = range(world_size) if gather_grad else (rank,)
                expected = sum((peer + 1) * chunks[peer].square().sum() for peer in ranks)
                (expected / world_size if scale_grad else expected).backward()
                ((rank + 1) * local.square().sum()).backward()
                assert torch.equal(full.grad, reference.grad)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_sequence_slice_preserves_default_and_local_gradients(tmp_path):
    mp.spawn(_slice_gradient_worker, args=(2, str(tmp_path / "slice-pg")), nprocs=2, join=True)


def _head_padding_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from vforge.distributed.ulysses import ulysses_attention

        torch.manual_seed(17)
        full_query = torch.randn(1, 16, 6, 4)
        full_key = torch.randn_like(full_query)
        full_value = torch.randn_like(full_query)
        seq_local = full_query.shape[1] // world_size
        seq_slice = slice(rank * seq_local, (rank + 1) * seq_local)
        query = full_query[:, seq_slice].clone().requires_grad_(True)
        key = full_key[:, seq_slice].clone().requires_grad_(True)
        value = full_value[:, seq_slice].clone().requires_grad_(True)

        def attention_fn(q, k, v, *, attn_mask, is_causal):
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            return output.transpose(1, 2)

        output = ulysses_attention(
            query,
            key,
            value,
            group=dist.group.WORLD,
            world_size=world_size,
            attention_fn=attention_fn,
        )

        reference_query = full_query.clone().requires_grad_(True)
        reference_key = full_key.clone().requires_grad_(True)
        reference_value = full_value.clone().requires_grad_(True)
        reference = attention_fn(
            reference_query,
            reference_key,
            reference_value,
            attn_mask=None,
            is_causal=False,
        )
        torch.testing.assert_close(output, reference[:, seq_slice])

        output.square().sum().backward()
        reference.square().sum().backward()
        torch.testing.assert_close(query.grad, reference_query.grad[:, seq_slice])
        torch.testing.assert_close(key.grad, reference_key.grad[:, seq_slice])
        torch.testing.assert_close(value.grad, reference_value.grad[:, seq_slice])

        gathered = {}
        for name, tensor in (
            ("output", output),
            ("query_grad", query.grad),
            ("key_grad", key.grad),
            ("value_grad", value.grad),
        ):
            parts = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(parts, tensor)
            gathered[name] = torch.cat(parts, dim=1)
        if rank == 0:
            torch.save(gathered, result_file)
        # Finish all ranks before any peer starts tearing down Gloo.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_sequence_head_round_trip_pads_and_trims_nondivisible_heads(
    monkeypatch,
) -> None:
    import vforge.distributed.ulysses as ulysses

    monkeypatch.setattr(
        ulysses,
        "sequence_all_to_all",
        lambda tensor, **_kwargs: tensor,
    )
    tensor = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    tensor.requires_grad_(True)

    head_sharded = ulysses.sequence_to_head(tensor, object(), world_size=3)
    restored = ulysses.head_to_sequence(
        head_sharded,
        object(),
        world_size=3,
        original_heads=5,
    )

    assert head_sharded.shape == (1, 6, 2, 3)
    torch.testing.assert_close(restored, tensor)
    restored.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


def test_ulysses_attention_usp2_and_usp8_match_with_head_padding(
    tmp_path,
) -> None:
    results = {}
    for usp_size in (2, 8):
        init_file = str(tmp_path / f"ulysses-head-padding-{usp_size}")
        result_file = str(tmp_path / f"ulysses-result-{usp_size}.pt")
        mp.spawn(
            _head_padding_worker,
            args=(usp_size, init_file, result_file),
            nprocs=usp_size,
            join=True,
        )
        results[usp_size] = torch.load(result_file, weights_only=True)

    for name, usp2_tensor in results[2].items():
        torch.testing.assert_close(usp2_tensor, results[8][name])


def test_ulysses_attention_hides_padded_heads_from_its_caller(monkeypatch) -> None:
    import vforge.distributed.ulysses as ulysses

    monkeypatch.setattr(
        ulysses,
        "sequence_all_to_all",
        lambda tensor, **_kwargs: tensor,
    )
    seen_shapes = {}

    def attention_fn(q, k, v, *, attn_mask, is_causal):
        seen_shapes.update(q=q.shape, k=k.shape, v=v.shape)
        assert attn_mask is None
        assert not is_causal
        return q

    query = torch.randn(1, 2, 5, 4)
    output = ulysses.ulysses_attention(
        query,
        query,
        query,
        group=object(),
        world_size=3,
        attention_fn=attention_fn,
    )

    assert seen_shapes == {
        "q": torch.Size([1, 6, 2, 4]),
        "k": torch.Size([1, 6, 2, 4]),
        "v": torch.Size([1, 6, 2, 4]),
    }
    assert output.shape == query.shape
    torch.testing.assert_close(output, query)


def test_ulysses_attention_preserves_grouped_query_ratio_when_padding(
    monkeypatch,
) -> None:
    import vforge.distributed.ulysses as ulysses

    monkeypatch.setattr(
        ulysses,
        "sequence_all_to_all",
        lambda tensor, **_kwargs: tensor,
    )
    seen_shapes = {}

    def attention_fn(q, k, v, *, attn_mask, is_causal):
        seen_shapes.update(q=q.shape, k=k.shape, v=v.shape)
        assert attn_mask is None
        assert not is_causal
        return q

    query = torch.randn(1, 2, 6, 4)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn_like(key)
    output = ulysses.ulysses_attention(
        query,
        key,
        value,
        group=object(),
        world_size=4,
        attention_fn=attention_fn,
    )

    assert seen_shapes == {
        "q": torch.Size([1, 8, 3, 4]),
        "k": torch.Size([1, 8, 1, 4]),
        "v": torch.Size([1, 8, 1, 4]),
    }
    assert output.shape == query.shape
    torch.testing.assert_close(output, query)


def test_wan_audio_ulysses_pads_nondivisible_heads(monkeypatch) -> None:
    import vforge.distributed.ulysses as ulysses
    import vforge.models.wan_taichu.audio as audio_module

    monkeypatch.setattr(
        ulysses,
        "sequence_all_to_all",
        lambda tensor, **_kwargs: tensor,
    )
    monkeypatch.setattr(audio_module.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        audio_module,
        "vforge_attention",
        lambda query, _key, _value, *, is_causal: query,
    )
    attention = audio_module.AudioCrossAttention(dim=10, num_heads=5)

    output = audio_module._audio_attention_ulysses(
        attention,
        torch.randn(1, 2, 10),
        torch.randn(2, 4, 10),
        group=object(),
        world_size=3,
        num_latent_frames=2,
        spatial_tokens=3,
    )

    assert output.shape == (1, 2, 10)


def test_attention_ulysses_is_explicit_opt_in(monkeypatch) -> None:
    import vforge.distributed.parallel_state as parallel_state
    import vforge.distributed.ulysses as ulysses
    from vforge.ops import attention

    group = object()
    state = SimpleNamespace(
        config=SimpleNamespace(usp_size=2),
        usp_group=group,
        cp_enabled=False,
    )
    captured = {}

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: state)

    def fake_ulysses(query, key, value, **kwargs):
        captured.update(kwargs)
        return query

    monkeypatch.setattr(
        ulysses,
        "ulysses_attention",
        fake_ulysses,
    )
    query = torch.randn(1, 2, 4, 8)

    output = attention(
        query,
        query,
        query,
        ulysses=True,
    )

    assert output is query
    assert captured["group"] is group
    assert captured["world_size"] == 2


def test_attention_distributed_selects_context_parallel(monkeypatch) -> None:
    import vforge.distributed.context_parallel as context_parallel
    import vforge.distributed.parallel_state as parallel_state
    from vforge.ops import attention

    state = SimpleNamespace(
        config=SimpleNamespace(usp_size=1),
        cp_enabled=True,
    )
    captured = {}
    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: state)

    def fake_context_parallel(query, key, value, **kwargs):
        captured.update(kwargs)
        return query

    monkeypatch.setattr(
        context_parallel,
        "context_parallel_attention",
        fake_context_parallel,
    )
    query = torch.randn(1, 2, 4, 8)

    output = attention(query, query, query, distributed=True)

    assert output is query
    assert captured == {"attn_mask": None, "is_causal": False}
