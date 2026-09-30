"""Constrained scoring of a fixed set of answers (an enum) with a causal LM.

The result is a typed decision: the chosen answer, a probability for every
answer and a confidence score.

Answers often span several tokens (e.g. "bearish" -> ["bear", "ish"]). The
candidates are arranged in a token trie; at every trie node the model's
logits are passed to the decoder with the node's children as the allowed
set. A word's probability is the product of the decoder probabilities along
its path, so the candidate probabilities sum to one. When every word is a
single token this reduces to one masked softmax over the next token.
"""

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decoders import Decoder, restrict_and_renormalize

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODES = ("trie", "sequence")


@dataclass
class ScoreResult:
    """A typed decision: one answer from a fixed enum, with probabilities."""

    prompt: str
    answers: list[str]
    # Probability of each answer after the decoder; sums to one.
    probabilities: dict[str, float]
    # Probability of each full answer under the unrestricted model.
    unconstrained_probs: dict[str, float]
    tokens: dict[str, list[str]] = field(default_factory=dict)

    @property
    def decision(self) -> str:
        """The most probable answer."""
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        """Share of the unrestricted model's mass that falls on the allowed answers.

        Low confidence means the model would rather have said something else.
        """
        return sum(self.unconstrained_probs.values())

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "decision": self.decision,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class ConstrainedScorer:
    """Load a causal LM once and score candidate words for many prompts.

    Parameters
    ----------
    model_name : Hugging Face model id or local path.
    decoder : function ``(logits, allowed_ids) -> probs``; see ``decoders.py``.
    use_chat_template : wrap prompts in the model's chat template. Defaults to
        True when the tokenizer has one.
    system_prompt : optional system message when using the chat template.
    device : torch device for the model, e.g. "cpu" or "cuda".
    dtype : weights dtype, e.g. ``torch.float32`` or ``torch.bfloat16``.
        Logits are always cast to float32 on the CPU before the decoder runs.
    mode : "trie" applies the decoder token by token (default). "sequence"
        computes each word's full log-probability under the unrestricted
        model and passes the vector of word log-probs to the decoder, which
        then allows word prefixes like "buy" / "buy more".
    model, tokenizer : pass already-loaded objects instead of ``model_name``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        decoder: Decoder = restrict_and_renormalize,
        use_chat_template: bool | None = None,
        system_prompt: str | None = None,
        mode: str = "trie",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        model=None,
        tokenizer=None,
        validate_decoder: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = model or AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device).eval()
        self.device = device
        self.decoder = decoder
        self.mode = mode
        self.system_prompt = system_prompt
        has_template = getattr(self.tokenizer, "chat_template", None) is not None
        self.use_chat_template = has_template if use_chat_template is None else use_chat_template
        self.validate_decoder = validate_decoder

    # ------------------------------------------------------------------ inputs

    def context_ids(self, prompt: str, system_prompt: str | None = None) -> list[int]:
        """Token ids of everything the model sees before the answer word."""
        if not self.use_chat_template:
            return self.tokenizer.encode(prompt, add_special_tokens=True)
        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def word_ids(self, word: str, leading_space: bool | None = None) -> list[int]:
        """Token ids of a candidate word as it would follow the context.

        A raw completion ("The outlook is") is followed by " bullish", while a
        chat answer starts without a space, hence the default.
        """
        if leading_space is None:
            leading_space = not self.use_chat_template
        ids = self.tokenizer.encode((" " if leading_space else "") + word, add_special_tokens=False)
        if not ids:
            raise ValueError(f"word {word!r} encodes to no tokens")
        return ids

    # ----------------------------------------------------------------- scoring

    @torch.no_grad()
    def score(
        self,
        prompt: str,
        answers: list[str],
        system_prompt: str | None = None,
        leading_space: bool | None = None,
    ) -> ScoreResult:
        """Return a decision over ``answers`` (an enum) for ``prompt``."""
        words = list(answers)
        if len(set(words)) != len(words):
            raise ValueError(f"duplicate answers in {words}")
        ctx = self.context_ids(prompt, system_prompt)
        seqs = [self.word_ids(w, leading_space) for w in words]
        _check_distinct(words, seqs, allow_prefixes=self.mode == "sequence")

        # One padded forward pass: row i = context + word i. Right padding
        # leaves the logits at real positions untouched under a causal mask.
        rows = [ctx + s for s in seqs]
        width = max(map(len, rows))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            attention[i, : len(r)] = 1
        logits = self.model(
            input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device)
        ).logits.float().cpu()

        # step_logits[i][j] = logits predicting token j of word i.
        n_ctx = len(ctx)
        step_logits = [[logits[i, n_ctx - 1 + j] for j in range(len(s))] for i, s in enumerate(seqs)]
        unconstrained_logp = torch.tensor(
            [
                sum(torch.log_softmax(step_logits[i][j], -1)[t].item() for j, t in enumerate(s))
                for i, s in enumerate(seqs)
            ]
        )

        if self.mode == "sequence":
            probs = self._apply_decoder(unconstrained_logp, torch.arange(len(words)))
            word_probs = probs.tolist()
        else:
            word_probs = self._trie_probs(seqs, step_logits)

        return ScoreResult(
            prompt=prompt,
            answers=words,
            probabilities=dict(zip(words, word_probs)),
            unconstrained_probs=dict(zip(words, unconstrained_logp.exp().tolist())),
            tokens={w: self.tokenizer.convert_ids_to_tokens(s) for w, s in zip(words, seqs)},
        )

    def score_many(self, prompts: list[str], answers: list[str], **kwargs) -> list[ScoreResult]:
        return [self.score(p, answers, **kwargs) for p in prompts]

    @torch.no_grad()
    def chat_complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 32,
    ) -> str:
        """Generate an unrestricted greedy completion from the scoring context."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        context = self.context_ids(prompt, system_prompt)
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        completion_ids = output[0, input_ids.shape[1] :]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    # --------------------------------------------------------------- internals

    def _trie_probs(self, seqs: list[list[int]], step_logits) -> list[float]:
        # Children of each prefix, and one row that reaches that prefix.
        children: dict[tuple, set[int]] = {}
        owner: dict[tuple, tuple[int, int]] = {}
        for i, s in enumerate(seqs):
            for j, tok in enumerate(s):
                prefix = tuple(s[:j])
                children.setdefault(prefix, set()).add(tok)
                owner.setdefault(prefix, (i, j))

        node_probs = {}
        for prefix, kids in children.items():
            i, j = owner[prefix]
            allowed = torch.tensor(sorted(kids), dtype=torch.long)
            node_probs[prefix] = self._apply_decoder(step_logits[i][j], allowed)

        out = []
        for s in seqs:
            p = 1.0
            for j, tok in enumerate(s):
                p *= node_probs[tuple(s[:j])][tok].item()
            out.append(p)
        return out

    def _apply_decoder(self, logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        probs = self.decoder(logits.clone(), allowed)
        if self.validate_decoder:
            _validate(probs, allowed, logits.shape[-1])
        return probs


def _check_distinct(words, seqs, allow_prefixes: bool) -> None:
    for a in range(len(seqs)):
        for b in range(len(seqs)):
            if a == b:
                continue
            sa, sb = seqs[a], seqs[b]
            if sa == sb:
                raise ValueError(f"{words[a]!r} and {words[b]!r} tokenize identically")
            if not allow_prefixes and len(sa) < len(sb) and sb[: len(sa)] == sa:
                raise ValueError(
                    f"{words[a]!r} is a token prefix of {words[b]!r}; the trie cannot tell "
                    "where the shorter word ends. Use mode='sequence' or rephrase a word."
                )


def _validate(probs: torch.Tensor, allowed: torch.Tensor, size: int) -> None:
    if probs.shape != (size,):
        raise ValueError(f"decoder returned shape {tuple(probs.shape)}, expected ({size},)")
    if torch.isnan(probs).any() or (probs < 0).any():
        raise ValueError("decoder returned negative or NaN probabilities")
    outside = probs.clone()
    outside[allowed] = 0
    if outside.sum() > 1e-6:
        raise ValueError("decoder put probability mass on tokens outside allowed_ids")
    if abs(probs.sum().item() - 1.0) > 1e-4:
        raise ValueError(f"decoder probabilities sum to {probs.sum().item():.6f}, not 1")
