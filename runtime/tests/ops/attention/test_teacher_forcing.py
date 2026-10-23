import pytest
import torch
import torch.nn.functional as F

from vforge.ops._attention import dispatch as attention
from vforge.ops._attention.teacher_forcing import (
    TeacherForcingSparsity,
    _tile_matrices,
    fa4_tile_size,
    frame_block_ids,
    teacher_forcing_dense_mask,
    teacher_forcing_sparsity_stats,
)


def _reference(q, k, v, spec):
    mask = teacher_forcing_dense_mask(spec, device=q.device)
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask).transpose(1, 2)


def _inputs(spec, *, heads=4, head_dim=64, seed=0):
    torch.manual_seed(seed)
    shape = (1, spec.seqlen, heads, head_dim)
    return tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------
def test_frame_blocks_give_the_first_frame_its_own_block():
    spec = TeacherForcingSparsity(num_latent_frames=7, tokens_per_frame=4, frame_per_block=3)
    assert frame_block_ids(spec).tolist() == [0, 1, 1, 1, 2, 2, 2]
    assert spec.num_blocks == 3


def test_uniform_blocks_drop_the_first_frame_exception():
    spec = TeacherForcingSparsity(
        num_latent_frames=6,
        tokens_per_frame=4,
        frame_per_block=3,
        first_frame_own_block=False,
    )
    assert frame_block_ids(spec).tolist() == [0, 0, 0, 1, 1, 1]
    assert spec.num_blocks == 2


def test_noisy_query_never_sees_its_own_clean_answer():
    spec = TeacherForcingSparsity(num_latent_frames=4, tokens_per_frame=2, frame_per_block=1)
    mask = teacher_forcing_dense_mask(spec)[0, 0]
    s = spec.tokens_per_half
    blocks = frame_block_ids(spec).repeat_interleave(spec.tokens_per_frame)

    for q in range(s, 2 * s):
        for k in range(s):
            same_or_later = blocks[k] >= blocks[q - s]
            assert bool(mask[q, k]) is not same_or_later or q - s == k


def test_clean_history_is_block_causal_and_noisy_stays_in_its_block():
    spec = TeacherForcingSparsity(num_latent_frames=4, tokens_per_frame=2, frame_per_block=1)
    mask = teacher_forcing_dense_mask(spec)[0, 0]
    s = spec.tokens_per_half
    blocks = frame_block_ids(spec).repeat_interleave(spec.tokens_per_frame)

    for q in range(s):
        for k in range(s):
            assert bool(mask[q, k]) == bool(blocks[k] <= blocks[q])
    for q in range(s, 2 * s):
        for k in range(s, 2 * s):
            same_block = blocks[k - s] == blocks[q - s]
            assert bool(mask[q, k]) == bool(same_block)


def test_dense_mask_refuses_the_production_shape():
    spec = TeacherForcingSparsity(num_latent_frames=31, tokens_per_frame=880, frame_per_block=3)
    with pytest.raises(ValueError, match="refusing to materialise"):
        teacher_forcing_dense_mask(spec)


# --------------------------------------------------------------------------
# Tile classification
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec",
    [
        TeacherForcingSparsity(num_latent_frames=7, tokens_per_frame=128, frame_per_block=3),
        TeacherForcingSparsity(num_latent_frames=5, tokens_per_frame=96, frame_per_block=2),
        TeacherForcingSparsity(
            num_latent_frames=6,
            tokens_per_frame=64,
            frame_per_block=2,
            first_frame_own_block=False,
        ),
        TeacherForcingSparsity(
            num_latent_frames=9,
            tokens_per_frame=64,
            frame_per_block=3,
            first_frame_own_block=False,
            local_attn_size=6,
            sink_size=3,
        ),
        TeacherForcingSparsity(
            num_latent_frames=31,
            tokens_per_frame=4,
            frame_per_block=3,
            first_frame_own_block=True,
            local_attn_size=31,
            sink_size=10,
        ),
    ],
)
def test_tile_classification_matches_the_dense_predicate(spec):
    tile = fa4_tile_size(spec.seqlen)
    tile_m, tile_n = tile
    partial, full = _tile_matrices(spec, tile)
    mask = teacher_forcing_dense_mask(spec)[0, 0]

    for i in range(partial.shape[0]):
        for j in range(partial.shape[1]):
            block = mask[i * tile_m : (i + 1) * tile_m, j * tile_n : (j + 1) * tile_n]
            whole = block.shape == (tile_m, tile_n)
            expect_full = bool(block.all()) and whole
            expect_any = bool(block.any())
            assert bool(full[i, j]) == expect_full
            assert bool(partial[i, j]) == (expect_any and not expect_full)


def test_stats_report_the_skipped_tiles():
    spec = TeacherForcingSparsity(num_latent_frames=31, tokens_per_frame=880, frame_per_block=3)
    stats = teacher_forcing_sparsity_stats(spec)
    assert stats["visited_tile_pairs"] + stats["skipped_tile_pairs"] == stats["total_tile_pairs"]
    # The whole point of the exercise: most of the doubled sequence is dead.
    assert stats["density"] < 0.3


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
@pytest.mark.gpu(gpus=1)
@pytest.mark.usefixtures("fa4_backend")
def test_teacher_forcing_matches_dense_sdpa():
    spec = TeacherForcingSparsity(num_latent_frames=7, tokens_per_frame=128, frame_per_block=3)
    q, k, v = _inputs(spec)
    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    out = attention.unified_attention(q, k, v, sparsity=spec)
    out_ref = _reference(q_ref, k_ref, v_ref, spec)
    torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=8e-3)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    for got, want in ((q, q_ref), (k, k_ref), (v, v_ref)):
        torch.testing.assert_close(got.grad, want.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu(gpus=1)
@pytest.mark.usefixtures("fa4_backend")
def test_sdpa_fallback_matches_the_kernel():
    spec = TeacherForcingSparsity(num_latent_frames=5, tokens_per_frame=64, frame_per_block=2)
    q, k, v = _inputs(spec, seed=7)

    kernel = attention.unified_attention(q, k, v, sparsity=spec)
    fallback = attention.unified_attention(q, k, v, sparsity=spec, native_mode=True)
    torch.testing.assert_close(kernel, fallback, rtol=2e-2, atol=8e-3)


def test_sparsity_rejects_a_second_mask():
    spec = TeacherForcingSparsity(num_latent_frames=2, tokens_per_frame=4, frame_per_block=1)
    q = torch.zeros(1, spec.seqlen, 1, 8)
    with pytest.raises(ValueError, match="describes the whole mask"):
        attention.unified_attention(
            q, q, q, attn_mask=torch.ones(1, 1, dtype=torch.bool), sparsity=spec
        )
    with pytest.raises(ValueError, match="describes the whole mask"):
        attention.unified_attention(q, q, q, is_causal=True, sparsity=spec)


def test_sparsity_rejects_a_sequence_it_does_not_describe():
    spec = TeacherForcingSparsity(num_latent_frames=2, tokens_per_frame=4, frame_per_block=1)
    q = torch.zeros(1, spec.seqlen + 1, 1, 8)
    with pytest.raises(ValueError, match="sparsity describes"):
        attention.unified_attention(q, q, q, sparsity=spec)


def test_spec_validates_its_geometry():
    with pytest.raises(ValueError, match="num_latent_frames"):
        TeacherForcingSparsity(num_latent_frames=0, tokens_per_frame=4, frame_per_block=1)
    with pytest.raises(ValueError, match="tokens_per_frame"):
        TeacherForcingSparsity(num_latent_frames=2, tokens_per_frame=0, frame_per_block=1)
    with pytest.raises(ValueError, match="frame_per_block"):
        TeacherForcingSparsity(num_latent_frames=2, tokens_per_frame=4, frame_per_block=0)
    with pytest.raises(ValueError, match="local_attn_size"):
        TeacherForcingSparsity(
            num_latent_frames=6,
            tokens_per_frame=4,
            frame_per_block=3,
            local_attn_size=4,
        )
    with pytest.raises(ValueError, match="sink_size"):
        TeacherForcingSparsity(
            num_latent_frames=6,
            tokens_per_frame=4,
            frame_per_block=3,
            local_attn_size=6,
            sink_size=6,
        )


def test_first_frame_block_accepts_taichu_31_10_window():
    spec = TeacherForcingSparsity(
        num_latent_frames=148,
        tokens_per_frame=4,
        frame_per_block=3,
        first_frame_own_block=True,
        local_attn_size=31,
        sink_size=10,
    )
    assert spec.sink_blocks == 4
    assert spec.recent_blocks == 7
