import pytest
param = pytest.mark.parametrize

import torch
from simple_hierarchical_transformer import HierarchicalTransformer

@param('hierarchies', (
    1,
    (1, 2),
    (1, 2, 4),
    (1, 2, 4, 8)
))
def test_hierarchical_transformer(hierarchies):
    model = HierarchicalTransformer(
        num_tokens = 256,
        dim = 64,
        depth = 2,
        dim_head = 16,
        heads = 4,
        seq_len = 64,
        hierarchies = hierarchies
    )

    ids = torch.randint(0, 256, (2, 64))

    loss, _ = model(ids, return_loss = True)
    assert loss.item() > 0

    loss.backward()
