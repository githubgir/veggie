"""Tests on a tiny randomly initialised GPT-2, so no model download is needed."""

import math

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from constrained_llm import ConstrainedScorer, make_temperature_decoder, restrict_and_renormalize

VOCAB = ["<pad>", "<unk>", "the", "stock", "is", "buy", "sell", "strong", "hold", "outlook", "?"]


@pytest.fixture(scope="module")
def tiny():
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(VOCAB), n_positions=32, n_embd=16, n_layer=2, n_head=2))
    return model, tokenizer


def make(tiny, **kw):
    model, tokenizer = tiny
    return ConstrainedScorer(model=model, tokenizer=tokenizer, use_chat_template=False, **kw)


def next_logits(tiny, ids):
    model, _ = tiny
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1]


def test_single_token_words_match_manual_masked_softmax(tiny):
    r = make(tiny).score("the stock is", ["buy", "sell", "hold"])
    logits = next_logits(tiny, [2, 3, 4])
    p = torch.softmax(logits, -1)[[5, 6, 8]]
    expected = (p / p.sum()).tolist()
    assert [r.probabilities[w] for w in ["buy", "sell", "hold"]] == pytest.approx(expected, abs=1e-6)
    assert r.confidence == pytest.approx(p.sum().item(), abs=1e-6)
    assert r.decision == ["buy", "sell", "hold"][int(p.argmax())]
    d = r.to_dict()
    assert d["decision"] == r.decision and d["probabilities"] == r.probabilities


def test_multi_token_words_follow_trie(tiny):
    words = ["buy", "strong buy", "strong sell"]
    r = make(tiny).score("the outlook is", words)
    assert sum(r.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    ctx = [2, 9, 4]
    first = torch.softmax(next_logits(tiny, ctx), -1)[[5, 7]]
    first = first / first.sum()  # P(buy), P(strong)
    second = torch.softmax(next_logits(tiny, ctx + [7]), -1)[[5, 6]]
    second = second / second.sum()  # P(buy | strong), P(sell | strong)
    assert r.probabilities["buy"] == pytest.approx(first[0].item(), abs=1e-6)
    assert r.probabilities["strong buy"] == pytest.approx((first[1] * second[0]).item(), abs=1e-6)
    assert r.probabilities["strong sell"] == pytest.approx((first[1] * second[1]).item(), abs=1e-6)


def test_batching_with_padding_does_not_change_scores(tiny):
    # "hold" is one token, so its row is padded next to the two-token words.
    r = make(tiny).score("the stock is", ["strong sell", "hold", "strong buy"])
    ctx = [2, 3, 4]
    p_hold = torch.softmax(next_logits(tiny, ctx), -1)[8].item()
    p_strong_sell = (
        torch.softmax(next_logits(tiny, ctx), -1)[7] * torch.softmax(next_logits(tiny, ctx + [7]), -1)[6]
    ).item()
    assert r.unconstrained_probs["hold"] == pytest.approx(p_hold, abs=1e-6)
    assert r.unconstrained_probs["strong sell"] == pytest.approx(p_strong_sell, abs=1e-6)


def test_sequence_mode_renormalises_word_likelihoods(tiny):
    words = ["strong", "strong buy", "hold"]
    r = make(tiny, mode="sequence").score("the stock is", words)
    total = sum(r.unconstrained_probs.values())
    for w in words:
        assert r.probabilities[w] == pytest.approx(r.unconstrained_probs[w] / total, abs=1e-6)


def test_trie_mode_rejects_prefix_words(tiny):
    with pytest.raises(ValueError, match="prefix"):
        make(tiny).score("the stock is", ["strong", "strong buy"])


def test_temperature_decoder_sharpens(tiny):
    base = make(tiny).score("the stock is", ["buy", "sell"]).probabilities
    sharp = make(tiny, decoder=make_temperature_decoder(0.1)).score("the stock is", ["buy", "sell"]).probabilities
    top = max(base, key=base.get)
    assert sharp[top] > base[top]


def test_custom_decoder_is_validated(tiny):
    def leaky(logits, allowed_ids):
        return torch.softmax(logits, -1)  # forgets to restrict

    with pytest.raises(ValueError, match="outside allowed_ids"):
        make(tiny, decoder=leaky).score("the stock is", ["buy", "sell"])


def test_uniform_custom_decoder(tiny):
    def uniform(logits, allowed_ids):
        p = torch.zeros_like(logits)
        p[allowed_ids] = 1.0 / len(allowed_ids)
        return p

    r = make(tiny, decoder=uniform).score("the stock is", ["buy", "sell", "strong buy", "strong sell"])
    assert r.probabilities["buy"] == pytest.approx(1 / 3)
    assert r.probabilities["strong sell"] == pytest.approx(1 / 6)


def test_user_decoder_matches_builtin(tiny):
    from my_decoder import decoder

    logits = torch.randn(len(VOCAB))
    allowed = torch.tensor([3, 5, 7])
    assert torch.allclose(decoder(logits, allowed), restrict_and_renormalize(logits, allowed), atol=1e-6)
    assert math.isclose(decoder(logits, allowed).sum().item(), 1.0, abs_tol=1e-6)
