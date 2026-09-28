"""Score a small, fixed set of answer words with a local causal LLM.

The model only produces next-token logits; a user-supplied *decoder* turns
those logits into probabilities restricted to the allowed tokens.
"""

from .decoders import Decoder, make_temperature_decoder, restrict_and_renormalize
from .scorer import DEFAULT_MODEL, ConstrainedScorer, ScoreResult

__all__ = [
    "DEFAULT_MODEL",
    "ConstrainedScorer",
    "Decoder",
    "ScoreResult",
    "make_temperature_decoder",
    "restrict_and_renormalize",
]
