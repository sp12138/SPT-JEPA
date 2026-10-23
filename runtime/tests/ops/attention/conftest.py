import pytest


@pytest.fixture
def fa4_backend():
    """Require FA4 kernels; missing dependencies must fail selected GPU tests."""
    from vforge.ops._attention import dispatch as attention

    attention.set_flash_attention_backend("flash_attention_4")
    assert attention.get_flash_attention_capability()["FLASH_ATTN_4_AVAILABLE"], (
        "GPU attention tests require the FA4 runtime"
    )
