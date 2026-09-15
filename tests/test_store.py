import unittest

from mpm.store import MemoryStore
from mpm.types import Action, Op, PayloadError


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.store.ensure_session("s", "scenario", "user")

    def tearDown(self):
        self.store.close()

    def test_update_preserves_revisions_and_delete_is_soft(self):
        mid = self.store.write_memory("v1", session_id="s")
        self.assertEqual(self.store.update_memory(mid, "v2", session_id="s"), 2)
        self.assertEqual([r["content"] for r in self.store.revisions(mid)], ["v1", "v2"])
        self.store.delete_memory(mid, session_id="s")
        self.assertEqual(self.store.get_memory(mid)["status"], "tombstoned")
        self.assertEqual(self.store.get_memory(mid)["content"], "v2")
        with self.assertRaises(PayloadError):
            self.store.update_memory(mid, "v3", session_id="s")

    def test_link_and_compact_retain_provenance(self):
        a = self.store.write_memory("a", session_id="s")
        b = self.store.write_memory("b", session_id="s")
        self.store.link_memories(a, b, "supports", 0.8, session_id="s")
        self.assertEqual(len(self.store.links_for(a)), 1)
        result = self.store.compact([a, b], "concat", session_id="s")
        self.assertEqual(self.store.get_memory(a)["status"], "compacted")
        self.assertEqual(self.store.get_memory(b)["status"], "compacted")
        self.assertEqual(self.store.get_memory(result)["status"], "active")
        members = self.store.conn.execute(
            "SELECT source_memory_id FROM compact_members ORDER BY source_memory_id"
        ).fetchall()
        self.assertEqual({r[0] for r in members}, {a, b})

    def test_checkpoint_rejection_never_activates_candidate(self):
        self.store.add_checkpoint("base", "base", status="active", gate_approved=True)
        self.store.activate_checkpoint("base")
        self.store.add_checkpoint("candidate", "candidate")
        self.store.reject_checkpoint("candidate", note="failed gate")
        self.assertEqual(self.store.get_active_checkpoint()["version"], "base")
        status = {r["version"]: r["status"] for r in self.store.list_checkpoints()}
        self.assertEqual(status["candidate"], "rejected")
        with self.assertRaises(PayloadError):
            self.store.activate_checkpoint("candidate")
        self.store.rollback_checkpoint("base", note="explicit rollback")
        self.assertEqual(self.store.get_active_checkpoint()["version"], "base")

    def test_only_gate_approved_candidate_can_be_promoted(self):
        self.store.add_checkpoint("base", "base", status="active", gate_approved=True)
        self.store.add_checkpoint("unapproved", "unapproved")
        with self.assertRaises(PayloadError):
            self.store.activate_checkpoint("unapproved")
        self.store.add_checkpoint("approved", "approved", gate_approved=True)
        self.store.activate_checkpoint("approved")
        self.assertEqual(self.store.get_active_checkpoint()["version"], "approved")
        self.assertEqual(
            {r["version"]: r["status"] for r in self.store.list_checkpoints()}["base"],
            "retired",
        )

    def test_invalid_outcome_is_rejected(self):
        with self.assertRaises(PayloadError):
            self.store.record_outcome("s", retrieval_id=None, kind="positive", value=-1, confidence=1)
        with self.assertRaises(PayloadError):
            self.store.record_outcome("s", retrieval_id=None, kind="negative", value=-1, confidence=2)

    def test_content_reference_is_materialized_only_by_executor(self):
        action = Action(op=Op.WRITE.value, payload={"content_ref": "observation.content"})
        with self.assertRaises(PayloadError):
            self.store.apply_action(action, session_id="s")
        result = self.store.apply_action(
            action,
            session_id="s",
            observation={"content": "private fact", "scope": "user"},
        )
        self.assertEqual(self.store.get_memory(result["memory_id"])["content"], "private fact")


if __name__ == "__main__":
    unittest.main()
