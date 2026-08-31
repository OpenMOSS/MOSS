import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FinetuneCliTest(unittest.TestCase):
    def test_accepts_fractional_warmup_rate(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "finetune_moss.py"),
                "--warmup_rates",
                "0.1",
                "--help",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
