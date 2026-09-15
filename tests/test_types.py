import unittest

from mpm.types import (
    ALL_OPS,
    Action,
    Op,
    PayloadError,
    action_from_dict,
    validate_action,
    validate_payload,
)


class TestOps(unittest.TestCase):
    def test_all_six_ops(self):
        self.assertEqual(ALL_OPS, {"WRITE", "UPDATE", "DELETE", "LINK", "COMPACT", "NOOP"})

    def test_validate_write(self):
        self.assertEqual(validate_payload(Op.WRITE.value, {"content": "x"}), [])
        self.assertEqual(validate_payload(Op.WRITE.value, {"content_ref": "observation.content"}), [])
        self.assertTrue(validate_payload(Op.WRITE.value, {}))
        self.assertTrue(validate_payload(Op.WRITE.value, {"content": ""}))

    def test_validate_update(self):
        self.assertEqual(validate_payload(Op.UPDATE.value, {"memory_id": "m1", "content": "y"}), [])
        self.assertEqual(
            validate_payload(Op.UPDATE.value, {"memory_id": "m1", "content_ref": "observation.content"}), []
        )
        self.assertTrue(validate_payload(Op.UPDATE.value, {"memory_id": "m1"}))

    def test_validate_delete(self):
        self.assertEqual(validate_payload(Op.DELETE.value, {"memory_id": "m1"}), [])
        self.assertTrue(validate_payload(Op.DELETE.value, {}))

    def test_validate_link(self):
        self.assertEqual(
            validate_payload(Op.LINK.value, {"source_id": "a", "target_id": "b", "kind": "rel", "weight": 0.5}),
            [],
        )
        self.assertTrue(
            validate_payload(Op.LINK.value, {"source_id": "a", "target_id": "a", "kind": "rel", "weight": 0.5})
        )
        self.assertTrue(
            validate_payload(Op.LINK.value, {"source_id": "a", "target_id": "b", "kind": "rel", "weight": 5.0})
        )

    def test_validate_compact(self):
        self.assertEqual(validate_payload(Op.COMPACT.value, {"memory_ids": ["a", "b"], "strategy": "concat"}), [])
        self.assertTrue(validate_payload(Op.COMPACT.value, {"memory_ids": ["a"], "strategy": "concat"}))
        self.assertTrue(validate_payload(Op.COMPACT.value, {"memory_ids": ["a", "a"], "strategy": "concat"}))

    def test_validate_noop(self):
        self.assertEqual(validate_payload(Op.NOOP.value, {"reason": "duplicate"}), [])
        self.assertTrue(validate_payload(Op.NOOP.value, {"observed": "not-a-dict"}))

    def test_validate_action_confidence(self):
        a = Action(op=Op.WRITE.value, payload={"content": "x"}, confidence=2.0)
        self.assertTrue(validate_action(a))

    def test_action_roundtrip(self):
        a = Action(op=Op.LINK.value, target="t", payload={"source_id": "a"}, confidence=0.5)
        b = action_from_dict(a.to_dict())
        self.assertEqual(b.op, a.op)
        self.assertEqual(b.target, "t")
        self.assertEqual(b.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
