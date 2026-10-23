import torch
from torch import nn

from vforge.utils.comparison import (
    batch_sha256,
    local_grad_norm,
    module_param_hash,
    rng_sha256,
    step_snapshot_json,
)


def test_batch_hash_is_stable_across_dict_order():
    left = {"b": torch.tensor([2]), "a": [torch.tensor([1]), "value"]}
    right = {"a": [torch.tensor([1]), "value"], "b": torch.tensor([2])}

    assert batch_sha256(left) == batch_sha256(right)


def test_gradient_and_parameter_helpers_accept_modules_and_iterators():
    module = nn.Linear(2, 1, bias=False)
    module(torch.ones(1, 2)).sum().backward()

    assert local_grad_norm(module) == local_grad_norm(module.parameters())
    assert module_param_hash(module) == module_param_hash(module)


def test_rng_hash_and_json_are_stable():
    module = nn.Linear(1, 1)
    torch.manual_seed(7)
    first = rng_sha256()
    torch.manual_seed(7)

    assert rng_sha256() == first
    snapshot = step_snapshot_json(module, torch.tensor(1.0), extra={"b": 2})
    assert snapshot.startswith('{"b":2,"grad_norm":')
