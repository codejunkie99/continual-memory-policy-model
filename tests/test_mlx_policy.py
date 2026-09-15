import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mpm.baseline import BaselinePolicy
from mpm.benchmark import FakeClock
from mpm.credit import memory_credit_summary
from mpm.live_eval import default_trajectory, run_live_evaluation
from mpm.mlx_policy import FakeMLXBackend, MLXPolicy, parse_op, tool_call_text
from mpm.store import MemoryStore
from mpm.types import Action, Op, PayloadError


def _always(op: str):
    def decide(observation_summary):
        return tool_call_text(op)

    return decide


class TestFakeBackendPrivacy(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.store.ensure_session("s", "scenario", "user")

    def tearDown(self):
        self.store.close()

    def test_prompt_never_contains_raw_content_or_extra_features(self):
        # Distinct canary values that do not trip the crude PII heuristics, so
        # the backend is actually consulted and the prompt can be inspected.
        secret_content = "CANARY-CONTENT-7q2x9v4m"
        secret_extra = "CANARY-EXTRA-3k8p5n1z"
        backend = FakeMLXBackend(decide=_always(Op.WRITE.value))
        policy = MLXPolicy(backend)

        action = policy.decide(
            {
                "content": secret_content,
                "scope": "private-scope",
                "key": "private-key",
                "extra_features": {"malicious": secret_extra},
            },
            self.store,
        )

        serialized_prompts = json.dumps(policy.prompts, sort_keys=True)
        self.assertNotIn(secret_content, serialized_prompts)
        self.assertNotIn(secret_extra, serialized_prompts)
        self.assertNotIn("private-scope", serialized_prompts)
        self.assertNotIn("private-key", serialized_prompts)
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(action.op, Op.WRITE.value)
        self.assertEqual(action.payload.get("content_ref"), "observation.content")
        self.assertNotIn("content", action.payload)

    def test_harmful_content_is_blocked_before_backend(self):
        backend = FakeMLXBackend(decide=_always(Op.WRITE.value))
        policy = MLXPolicy(backend)

        action = policy.decide({"content": "contact me at alice@corp.com", "scope": "work"}, self.store)

        self.assertEqual(action.op, Op.NOOP.value)
        self.assertEqual(backend.prompts, [])  # backend never consulted
        self.store.apply_action(action, session_id="s", observation={"content": "contact me at alice@corp.com"})
        self.assertEqual(len(self.store.active_memories()), 0)

    def test_executor_rejects_harmful_direct_and_referenced_content(self):
        with self.assertRaises(PayloadError):
            self.store.apply_action(
                Action(Op.WRITE.value, payload={"content": "email me at alice@corp.com"}),
                session_id="s",
            )
        with self.assertRaises(PayloadError):
            self.store.apply_action(
                Action(Op.WRITE.value, payload={"content_ref": "observation.content"}),
                session_id="s",
                observation={"content": "password=hunter2"},
            )
        self.assertEqual(self.store.active_memories(), [])

    def test_model_receives_safe_semantic_signals_and_aliases(self):
        existing = self.store.write_memory("the release is June", scope="work", session_id="s")
        backend = FakeMLXBackend(decide=_always(Op.NOOP.value))
        policy = MLXPolicy(backend)
        policy.decide(
            {
                "content": "the release is June",
                "scope": "work",
                "intent": "delete",
                "context_ids": [existing],
            },
            self.store,
        )
        features = backend.prompts[0]["features"]
        self.assertEqual(features["requested_op"], "DELETE")
        self.assertEqual(features["context_count"], 1)
        self.assertTrue(features["exact_duplicate"])
        self.assertEqual(features["candidate_ids"], ["candidate_0"])
        self.assertNotIn(existing, json.dumps(backend.prompts))


class TestNativeParser(unittest.TestCase):
    def test_accepts_native_single_or_double_quote_tool_call(self):
        self.assertEqual(parse_op(tool_call_text(Op.LINK.value)), Op.LINK.value)
        self.assertEqual(
            parse_op("<|tool_call_start|>[memory_action(op='COMPACT', payload={})]<|tool_call_end|>"),
            Op.COMPACT.value,
        )

    def test_rejects_prose_or_extra_calls(self):
        self.assertIsNone(parse_op("I think you should WRITE this"))
        self.assertIsNone(parse_op(tool_call_text(Op.WRITE.value) + " trailing text"))


class TestFakeBackendLiveTrajectory(unittest.TestCase):
    def test_harmful_trajectory_never_writes_harmful_content(self):
        clock = FakeClock()
        store = MemoryStore(":memory:", clock=clock)
        backend = FakeMLXBackend(decide=_always(Op.WRITE.value))
        policy = MLXPolicy(backend)

        metrics = run_live_evaluation(store, policy, clock=clock)

        self.assertEqual(metrics["harmful_memory_rate"], 0.0)
        self.assertEqual(metrics["harmful_write_attempt_rate"], 0.0)
        self.assertEqual(metrics["harmful_content_stored"], 0)
        self.assertEqual(metrics["privacy_leakage"], 0.0)
        store.close()

    def test_scripted_backend_scores_full_accuracy_and_reconciles_credit(self):
        clock = FakeClock()
        store = MemoryStore(":memory:", clock=clock)
        trajectory = default_trajectory()
        ops = iter([step.expected for step in trajectory if not step.harmful])
        backend = FakeMLXBackend(decide=lambda summary: tool_call_text(next(ops)))
        policy = MLXPolicy(backend)

        metrics = run_live_evaluation(store, policy, trajectory=trajectory, clock=clock)

        self.assertEqual(metrics["operation_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["structured_output_validity"], 1.0)
        self.assertEqual(metrics["action_execution_validity"], 1.0)
        self.assertEqual(metrics["attributions"], 5)
        self.assertGreater(metrics["downstream_utility"], 0.0)

        rewards = {m["memory_id"]: m["total_reward"] for m in memory_credit_summary(store)}
        stale = next(m for m in store.all_memories() if "March 1st" in m["content"])
        self.assertLess(rewards[stale["memory_id"]], 0.0)
        store.close()

    def test_invalid_output_falls_back_to_noop(self):
        backend = FakeMLXBackend(decide=lambda summary: "garbage without an operation")
        policy = MLXPolicy(backend)
        store = MemoryStore(":memory:")

        action = policy.decide({"content": "a normal fact", "scope": "work"}, store)

        self.assertFalse(policy.last_structured_valid)
        self.assertEqual(action.op, Op.NOOP.value)
        store.close()

    def test_baseline_policy_also_runs_live(self):
        clock = FakeClock()
        store = MemoryStore(":memory:", clock=clock)
        metrics = run_live_evaluation(store, BaselinePolicy(), clock=clock)
        for key in (
            "operation_accuracy",
            "macro_f1",
            "downstream_utility",
            "harmful_memory_rate",
            "privacy_leakage",
            "structured_output_validity",
            "latency_s",
        ):
            self.assertIn(key, metrics)
        store.close()


class TestLazyImports(unittest.TestCase):
    def test_importing_module_does_not_import_mlx(self):
        src = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys; "
            "import mpm.mlx_policy; "
            "assert 'mlx' not in sys.modules, 'mlx imported eagerly'; "
            "assert 'mlx_vlm' not in sys.modules, 'mlx_vlm imported eagerly'; "
            "print('lazy-ok')"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestLiveCli(unittest.TestCase):
    def test_live_evaluate_cli_with_fake_backend(self):
        from mpm.cli import main

        with tempfile.TemporaryDirectory() as td:
            rc = main(
                [
                    "live-evaluate",
                    "--db",
                    ":memory:",
                    "--policy",
                    "mlx",
                    "--backend",
                    "fake",
                    "--outdir",
                    td,
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads((Path(td) / "live-evaluation.json").read_text())
            self.assertEqual(report["policy"], "mlx")
            self.assertIn("operation_accuracy", report)
            self.assertFalse(report["promotion_gate"]["approved"])


if __name__ == "__main__":
    unittest.main()
