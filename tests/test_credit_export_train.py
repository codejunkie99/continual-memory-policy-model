import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mpm.credit import compute_attributions, reconcile_all
from mpm.export import assign_split, export_all
from mpm.store import MemoryStore
from mpm.train.config import TrainConfig
from mpm.train.dryrun import dry_run


class TestCredit(unittest.TestCase):
    def test_one_delayed_outcome_is_shared_across_retrievals(self):
        now = [40.0]
        store = MemoryStore(":memory:", clock=lambda: now[0])
        store.ensure_session("s", "launch", "u")
        a = store.write_memory("April 12", session_id="s")
        b = store.write_memory("March 1 stale", session_id="s")
        now[0] = 8000.0
        ra = store.record_retrieval("s", a, {}, rank=0, score=0.9)
        rb = store.record_retrieval("s", b, {}, rank=1, score=0.8)
        oid = store.record_outcome(
            "s",
            retrieval_id=None,
            retrieval_weights={ra: 0.75, rb: 0.25},
            kind="positive",
            value=1.0,
            confidence=0.8,
        )
        computed = compute_attributions(store, oid)
        self.assertEqual(len(computed), 2)
        self.assertGreater(computed[0]["reward"] + computed[1]["reward"], 0)
        self.assertLess(computed[0]["reward"] + computed[1]["reward"], 1.0)
        self.assertAlmostEqual(sum(x["audit"]["normalized_contribution"] for x in computed), 1.0)
        self.assertEqual(len(reconcile_all(store)), 2)
        self.assertEqual(len(reconcile_all(store)), 0)
        store.close()

    def test_positive_and_negative_repeated_retrievals_accumulate(self):
        store = MemoryStore(":memory:")
        store.ensure_session("s", "x", "u")
        mid = store.write_memory("fact", session_id="s")
        for value, kind in ((1.0, "positive"), (-0.5, "negative")):
            rid = store.record_retrieval("s", mid, {}, rank=0, score=0.8)
            store.record_outcome("s", retrieval_id=rid, kind=kind, value=value, confidence=1.0)
        self.assertEqual(len(reconcile_all(store)), 2)
        rewards = [r["reward"] for r in store.attribution_events_for_memory(mid)]
        self.assertTrue(any(r > 0 for r in rewards))
        self.assertTrue(any(r < 0 for r in rewards))
        store.close()


class TestExportAndTraining(unittest.TestCase):
    def test_default_export_excludes_raw_and_plain_hash(self):
        secret = "TOP-SECRET-user-value-4815162342"
        store = MemoryStore(":memory:")
        store.ensure_session("s", "private-scenario", "private-user")
        store.write_memory(
            secret,
            scope="private-scope",
            key="private-key",
            session_id="s",
            features={"malicious_extra": secret, "content_hash": hashlib.sha256(secret.encode()).hexdigest()[:16]},
        )
        with tempfile.TemporaryDirectory() as td:
            export_all(store, td)
            corpus = "\n".join(p.read_text() for p in Path(td).glob("*.jsonl"))
            self.assertNotIn(secret, corpus)
            self.assertNotIn(hashlib.sha256(secret.encode()).hexdigest()[:16], corpus)
            self.assertNotIn("private-user", corpus)
            self.assertNotIn("private-scenario", corpus)
            self.assertNotIn("private-scope", corpus)
            self.assertNotIn("private-key", corpus)
        store.close()

    def test_training_actions_use_content_reference_and_local_candidate_aliases(self):
        store = MemoryStore(":memory:")
        store.ensure_session("s", "scenario", "user")
        mid = store.write_memory("first", session_id="s")
        store.update_memory(mid, "second", session_id="s")
        with tempfile.TemporaryDirectory() as td:
            export_all(store, td, ratios=[("train", 1.0)])
            rows = [json.loads(line) for line in (Path(td) / "sft-train.jsonl").read_text().splitlines()]
            write = next(row for row in rows if row["label"] == "WRITE")
            update = next(row for row in rows if row["label"] == "UPDATE")
            self.assertIsNone(write["action"]["target"])
            self.assertEqual(write["action"]["payload"]["content_ref"], "observation.content")
            self.assertEqual(update["action"]["target"], "candidate_0")
            self.assertEqual(update["action"]["payload"]["memory_id"], "candidate_0")
            self.assertIn("candidate_0", update["features"]["candidate_ids"])
            self.assertNotIn(mid, json.dumps(rows))
        store.close()

    def test_unsafe_export_is_explicit_and_contains_raw(self):
        secret = "allowed-only-with-unsafe-flag"
        store = MemoryStore(":memory:")
        store.ensure_session("s", "scenario", "user")
        store.write_memory(secret, session_id="s")
        with tempfile.TemporaryDirectory() as td:
            export_all(store, td, allow_raw=True)
            corpus = "\n".join(p.read_text() for p in Path(td).glob("*.jsonl"))
            self.assertIn(secret, corpus)
        store.close()

    def test_split_is_deterministic(self):
        first = [assign_split(f"scenario-{i}", f"user-{i % 7}", seed="fixed") for i in range(100)]
        second = [assign_split(f"scenario-{i}", f"user-{i % 7}", seed="fixed") for i in range(100)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)

    def test_dry_run_emits_official_leap_shape(self):
        store = MemoryStore(":memory:")
        for i in range(30):
            sid = f"s{i}"
            store.ensure_session(sid, f"scenario-{i}", f"u-{i}")
            store.write_memory(f"synthetic fact {i}", session_id=sid)
        with tempfile.TemporaryDirectory() as td:
            export_all(store, td, seed="fixed")
            report = dry_run(td, TrainConfig(), outdir=td, seed="fixed")
            self.assertEqual(report["status"], "ok")
            yaml_text = (Path(td) / "leap-finetune.yaml").read_text()
            self.assertIn('training_type: "vlm_sft"', yaml_text)
            self.assertIn('extends: "DEFAULT_VLM_SFT"', yaml_text)
            self.assertIn('extends: "DEFAULT_VLM_LORA"', yaml_text)
            modal_yaml = (Path(td) / "leap-finetune-modal.yaml").read_text()
            self.assertIn('gpu: "H100"', modal_yaml)
            self.assertIn('output_volume: "mpm-lfm25-vl"', modal_yaml)
            self.assertTrue((Path(td) / "modal-upload-commands.txt").exists())
            rows = [json.loads(line) for line in (Path(td) / "leap-vlm-sft-train.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertIn("messages", rows[0])
            self.assertIsInstance(rows[0]["messages"][0]["content"], list)
            self.assertEqual(rows[0]["messages"][-1]["role"], "tool")
        store.close()


if __name__ == "__main__":
    unittest.main()
