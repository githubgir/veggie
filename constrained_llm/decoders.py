"""Decoder functions: map raw next-token logits to a restricted distribution.

A decoder has the signature

    decoder(logits: Tensor[V], allowed_ids: LongTensor[k]) -> Tensor[V]

``logits`` are the model's raw scores for every token in the vocabulary.
``allowed_ids`` are the only tokens that may be emitted at this step.
The returned tensor must be a probability vector over the full vocabulary
that is zero outside ``allowed_ids`` and sums to one.
"""

from typing import Callable

import torch

Decoder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def restrict_and_renormalize(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
    """Zero every token outside ``allowed_ids`` and rescale the rest to sum to one.

    Equivalent to p_i / sum_{j in allowed} p_j with p = softmax(logits).
    """
    masked = torch.full_like(logits, float("-inf"))
    masked[allowed_ids] = logits[allowed_ids]
    return torch.softmax(masked, dim=-1)


def make_temperature_decoder(temperature: float) -> Decoder:
    """Restrict-and-renormalize after dividing logits by ``temperature``.

    temperature < 1 sharpens the distribution, > 1 flattens it.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    def decoder(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
        return restrict_and_renormalize(logits / temperature, allowed_ids)

    return decoder
