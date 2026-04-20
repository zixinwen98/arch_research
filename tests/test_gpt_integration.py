import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from gpt2 import Config, GPT


def test_gpt_forward_backward_integration():
    torch.manual_seed(2)
    cfg = Config(
        vocab_size=120,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_mlp=128,
        sliding_window=4,
        max_seq_len=64,
        rope_base=10000.0,
        tie_output=True,
    )
    model = GPT(cfg)
    batch_size, seq_len = 3, 10
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    attn_mask = torch.ones((batch_size, seq_len), dtype=torch.int64)
    attn_mask[0, -2:] = 0
    attn_mask[1, -4:] = 0
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, seq_len)
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits = model(input_ids, attn_mask=attn_mask, position_ids=position_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    loss.backward()

    grad_norm = model.tok_emb.weight.grad.norm().item()
    assert grad_norm > 0.0


if __name__ == "__main__":
    test_gpt_forward_backward_integration()
    print("gpt integration ok")
