import io
import os
import sys
from contextlib import redirect_stdout


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tttlab.training.capo_ttt import main


def test_capo_training_smoke():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = main(
            [
                "--steps",
                "1",
                "--batch_entries",
                "4",
                "--window",
                "32",
                "--device",
                "cpu",
                "--tokenizer",
                "ws",
            ]
        )

    output = buffer.getvalue()
    assert rc == 0
    assert "Tokenizer: ws" in output
    assert "step 1/1" in output
    assert "Training run finished." in output


if __name__ == "__main__":
    test_capo_training_smoke()
    print("capo training smoke ok")
