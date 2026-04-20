import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from arch_research.models.gpt_ttt import Config, GPT_TTT


def test_gpt_ttt_chunked_forward_backward_integration():
    torch.manual_seed(0)
    cfg = Config(
        vocab_size=300,
        n_layers=2,
        n_heads=4,
        d_head=16,
        d_model=64,
        d_mlp=128,
        d_ttt=32,
        sliding_window=0,
        tie_output=False,
        use_mlp=True,
    )
    model = GPT_TTT(cfg)
    batch_size, seq_len = 1, 512
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    chunk = 128
    ttt_config = []
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        ttt_config.append(("apply_then_update", start, end))

    logits = model(
        input_ids,
        attn_mask_1d=attn_mask,
        position_ids=position_ids,
        ttt_config=ttt_config,
    )
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    loss.backward()

    total_grad = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad += float(param.grad.norm())
    assert total_grad > 0.0


if __name__ == "__main__":
    test_gpt_ttt_chunked_forward_backward_integration()
    print("gpt_ttt integration ok")
