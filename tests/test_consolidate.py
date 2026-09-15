import json
import tempfile
import unittest
from pathlib import Path

from mpm.consolidate import build_consolidation


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class TestConsolidation(unittest.TestCase):
    def test_current_plus_difficult_first_deduplicated_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current, replay, out = root / "current", root / "replay", root / "out"
            base = {"split": "train", "kind": "sft", "stratum": "easy", "label": "WRITE"}
            write_rows(current / "sft-train.jsonl", [base])
            write_rows(current / "dpo-train.jsonl", [])
            write_rows(current / "sft-val.jsonl", [{**base, "split": "val"}])
            write_rows(
                replay / "sft-train.jsonl",
                [base, {**base, "label": "NOOP", "stratum": "hard"}, {**base, "label": "DELETE", "stratum": "negative"}],
            )
            write_rows(replay / "dpo-train.jsonl", [])
            report = build_consolidation(current, [replay], out, max_replay_per_kind=1, seed="fixed")
            rows = [json.loads(line) for line in (out / "consolidated-sft.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "WRITE")
            self.assertEqual(rows[1]["stratum"], "negative")
            self.assertEqual(report["kinds"]["sft"]["historical_selected"], 1)
            self.assertFalse(report["auto_promote"])
            self.assertEqual(report["leap_sft"]["status"], "ready")
            self.assertEqual(report["leap_preference"]["status"], "not_ready")
            self.assertTrue((out / "leap-consolidation.yaml").exists())
            leap_rows = [json.loads(line) for line in (out / "leap-vlm-sft-train.jsonl").read_text().splitlines()]
            self.assertIn("messages", leap_rows[0])

    def test_rejects_non_train_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_rows(root / "current" / "sft-train.jsonl", [{"split": "val"}])
            write_rows(root / "current" / "dpo-train.jsonl", [])
            with self.assertRaises(ValueError):
                build_consolidation(root / "current", [], root / "out")


if __name__ == "__main__":
    unittest.main()
