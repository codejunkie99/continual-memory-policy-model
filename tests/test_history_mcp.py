import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mpm.baseline import BaselinePolicy
from mpm.history.build import assign_time_ordered_splits, build_dataset
from mpm.history.ingest import IngestConfig, ingest_sources
from mpm.history.staging import HistoryStagingStore
from mpm.mcp_server import MemoryService
from mpm.store import MemoryStore


class TestHistoryPipeline(unittest.TestCase):
    def test_streaming_ingest_and_private_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            codex = root / "codex"
            claude = root / "claude"
            brain = root / "brain"
            codex.mkdir()
            claude.mkdir()
            brain.mkdir()

            (codex / "session.jsonl").write_text(
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": 1000,
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "i prefer concise reports"}],
                        },
                    }
                )
                + "\nnot-json\n"
                + json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": 1001,
                        "payload": {
                            "type": "reasoning",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "private reasoning"}],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (claude / "history.jsonl").write_text(
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": 2000,
                        "message": {"content": "we always use snake_case names"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "user",
                        "timestamp": 2001,
                        "message": {
                            "content": [
                                {"type": "tool_result", "content": "i prefer leaked tool output"}
                            ]
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "attachment",
                        "timestamp": 2002,
                        "content": "i prefer this must not be read",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            brain_db = brain / "index.sqlite"
            conn = sqlite3.connect(brain_db)
            conn.execute(
                "CREATE TABLE events (event_id TEXT, payload_json TEXT, "
                "time_observed INTEGER, is_redacted INTEGER)"
            )
            conn.execute(
                "INSERT INTO events VALUES (?,?,?,0)",
                ("opaque", json.dumps({"summary": "the lesson learned was to test rollback"}), 3000),
            )
            conn.commit()
            conn.close()

            mpm_db = root / "source.db"
            source = MemoryStore(mpm_db)
            source.write_memory("the selected release channel is stable", scope="decision")
            source.close()

            staging = HistoryStagingStore(root / "staging.db", salt="x" * 32)
            report = ingest_sources(
                staging,
                codex_dir=codex,
                claude_dir=claude,
                brain_dir=brain,
                mpm_db=[mpm_db],
                cfg=IngestConfig(salt="x" * 32),
            )
            self.assertEqual(report.malformed, 1)
            self.assertEqual(report.accepted_by_source["codex"], 1)
            self.assertEqual(report.accepted_by_source["claude"], 1)
            self.assertEqual(report.accepted_by_source["brain"], 1)
            self.assertEqual(report.accepted_by_source["mpm"], 1)

            out = root / "dataset"
            manifest = build_dataset(staging, out, time_bucket=1.0)
            staging.close()
            self.assertNotIn("salt", manifest)
            self.assertTrue(manifest["privacy_checks"]["no_raw_text"])
            emitted = "\n".join(p.read_text() for p in out.glob("*.jsonl"))
            for raw in (
                "i prefer concise reports",
                "we always use snake_case names",
                "the lesson learned was to test rollback",
                "the selected release channel is stable",
            ):
                self.assertNotIn(raw, emitted)
            self.assertIn('"messages"', (out / "mlx-train.jsonl").read_text())

    def test_temporal_cohorts_never_move_backwards(self):
        records = [
            {"ts": float(i), "salted_hash": str(i)} for i in range(1, 11)
        ]
        assigned = assign_time_ordered_splits(records, time_bucket=1.0)
        order = {"train": 0, "val": 1, "test": 2}
        self.assertEqual(
            [order[split] for _, split in assigned],
            sorted(order[split] for _, split in assigned),
        )


class TestMemoryService(unittest.TestCase):
    def test_deterministic_scoped_search_and_delayed_feedback(self):
        store = MemoryStore(":memory:")
        release_id = store.write_memory("release channel is stable", scope="work")
        store.write_memory("reports should be concise", scope="writing")
        service = MemoryService(store, BaselinePolicy(), session_id="mcp-test")

        first = service.search("release", scope="work", limit=10)
        second = service.search("release", scope="work", limit=10)
        self.assertEqual([r["memory_id"] for r in first["results"]], [release_id])
        self.assertEqual(
            [r["memory_id"] for r in first["results"]],
            [r["memory_id"] for r in second["results"]],
        )
        self.assertEqual(service.search("unrelated-token")["results"], [])

        feedback = service.feedback(
            first["retrieval_ids"][0], kind="positive", value=1.0, confidence=1.0
        )
        self.assertTrue(feedback["ok"])
        self.assertEqual(feedback["attributions"][0]["memory_id"], release_id)
        store.close()

    def test_observe_reuses_write_safety_gate(self):
        store = MemoryStore(":memory:")
        service = MemoryService(store, BaselinePolicy(), session_id="mcp-safety")
        result = service.observe("my password is swordfish", scope="private")
        self.assertEqual(result["op"], "NOOP")
        self.assertEqual(store.active_memories(), [])
        store.close()


if __name__ == "__main__":
    unittest.main()
