"""
Module for utilities.
"""

import torch


def get_mask(input_ids, pad_idx):
    """
    Get the mask for the input sequences.

    The mask is made of two parts:
    - the padding tokens are masked (to ignore them in the loss computation)
    - the future tokens are masked (for the next token prediction)

    Args:
        input_ids (torch.LongTensor): tensor of input token indices, of shape (batch_size, seq_len).
        pad_idx (int): index of the padding token.
    Returns:
        mask (torch.LongTensor): mask for the self-attention, of shape (batch_size, seq_len, seq_len).
            0 in the mask means that the token is masked, 1 means that the token is not masked.
    """
    seq_len = input_ids.shape[-1]
    # Mask the padding tokens
    mask_pad = input_ids != pad_idx
    mask_pad = mask_pad.unsqueeze(1).expand(
        -1, seq_len, -1
    )  # (batch_size, seq_len, seq_len)

    # Mask the future tokens (for causal attention)
    mask_causal = torch.tril(
        torch.ones(seq_len, seq_len, device=input_ids.device)
    ).bool()

    # Combine the two masks
    mask = mask_pad & mask_causal

    return mask.to(int)
