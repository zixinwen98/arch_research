import io
import os
import sys
from contextlib import redirect_stdout


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arch_research.cli import train_main


def test_capo_training_smoke():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = train_main(
            [
                "--task",
                "capo",
                "--model",
                "gpt_ttt",
                "--steps",
                "1",
                "--batch-size",
                "4",
                "--window",
                "32",
                "--device",
                "cpu",
                "--task-arg",
                "tokenizer=ws",
                "--model-arg",
                "n_layers=1",
                "--model-arg",
                "n_heads=2",
                "--model-arg",
                "d_head=16",
                "--model-arg",
                "d_model=32",
                "--model-arg",
                "d_mlp=64",
                "--model-arg",
                "d_ttt=0",
                "--eval-batches",
                "1",
            ]
        )

    output = buffer.getvalue()
    assert rc == 0
    assert "step 1/1" in output
    assert "validation" in output
    assert "evaluation" in output


if __name__ == "__main__":
    test_capo_training_smoke()
    print("capo training smoke ok")
