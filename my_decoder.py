"""Your decoder. Edit ``decoder`` freely, then point experiments.yaml at it.

    logits      : torch.Tensor [vocab_size], raw model scores for the next token
    allowed_ids : torch.LongTensor [k], the only token ids you may output
    return      : torch.Tensor [vocab_size], probabilities, zero outside
                  allowed_ids and summing to 1 (the scorer checks this)
"""

import torch


def decoder(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
    probs = torch.zeros_like(logits)
    p = torch.softmax(logits, dim=-1)  # the model's full distribution
    probs[allowed_ids] = p[allowed_ids]  # keep only the allowed tokens
    return probs / probs.sum()  # rescale to sum to one


def sharp_decoder(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
    """Example variant: temperature 0.5 makes the model more decisive."""
    return decoder(logits / 0.5, allowed_ids)
