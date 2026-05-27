"""Minimal test for invoking monotonic_align"""

import torch
from monotonic_align import mask_from_lens

from ..utils import maximum_path


def test_maximum_path_JK_eg2():
    # Test copied from tests/test_max_path.py in https://github.com/resemble-ai/monotonic_align
    # Copyright (c) 2020 Jaehyeon Kim -- MIT License

    # We're copying it here to validate the clean integration of our ilt-monotonic-align
    # fork in this project.

    # Begin Example 2 copied from monotonic_align
    B = 16  # batch_size
    S = 45  # max_symbol_len
    T = 500  # max_mel_len
    M = 80  # num_mels

    symbol_embs = torch.randn(B, S, M)
    symbol_lens = torch.randint(1, S, size=[B])

    mels = torch.randn(B, T, M)
    mel_lens = torch.randint(1, T, size=[B])

    similarity = -(symbol_embs.unsqueeze(2) - mels.unsqueeze(1)).pow(2).sum(-1)
    mask_ST = mask_from_lens(similarity, symbol_lens, mel_lens)
    alignments = maximum_path(similarity, mask_ST)
    # End Example 2 copied from monotonic_align

    assert isinstance(alignments, torch.Tensor)
    assert alignments.size() == torch.Size([16, 45, 500])
