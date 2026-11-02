"""Durability receipts must never turn lost workers into successful experiments."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("control", SCRIPTS / "control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class WorkerReceiptTests(unittest.TestCase):
    def run_status(self, receipt, observed_birth):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "fixture"
            run.mkdir(parents=True)
            (run / "status.json").write_text(json.dumps(receipt))
            with patch.object(control, "process_birth", return_value=observed_birth):
                return control.status(argparse.Namespace(run="fixture"), root)

    def test_dead_worker_is_lost_not_success(self):
        status = self.run_status({"state": "running", "worker_pid": 100, "worker_birth": "123", "exit_code": None}, None)
        self.assertEqual(status["state"], "lost")
        self.assertIsNone(status["exit_code"])

    def test_reused_pid_is_lost_not_success(self):
        status = self.run_status({"state": "running", "worker_pid": 100, "worker_birth": "123", "exit_code": None}, "456")
        self.assertEqual(status["state"], "lost")

    def test_completed_receipt_survives_disconnected_worker(self):
        status = self.run_status({"state": "completed", "worker_pid": 100, "worker_birth": "123", "exit_code": 0}, None)
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["exit_code"], 0)

    def test_pilot_requires_matching_real_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "candidate"
            run.mkdir(parents=True)
            (run / "request.json").write_text(json.dumps({"kind": "fixtures"}))
            (run / "status.json").write_text(json.dumps({"state": "completed", "exit_code": 0}))
            (run / "summary.json").write_text(json.dumps({"complete": True, "passed": True,
                "cases": [{"action": action} for action in ("retain", "spill", "recompute")]}))
            (run / "environment.json").write_text(json.dumps({"artifact_completion_sha256": "hash",
                "model": {"revision": "revision"}}))
            with self.assertRaises(ValueError):
                control.successful_pretrained_smoke(root, "hash", "revision", "lock")
            (run / "request.json").write_text(json.dumps({"kind": "smoke", "dependency_lock_sha256": "lock"}))
            self.assertEqual(control.successful_pretrained_smoke(root, "hash", "revision", "lock"), "candidate")
            with self.assertRaises(ValueError):
                control.successful_pretrained_smoke(root, "other-hash", "revision", "lock")

    def test_pilot_rejects_smoke_from_another_dependency_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "candidate"
            run.mkdir(parents=True)
            (run / "status.json").write_text(json.dumps({"state": "completed", "exit_code": 0}))
            (run / "summary.json").write_text(json.dumps({"complete": True, "passed": True,
                "cases": [{"action": action} for action in ("retain", "spill", "recompute")]}))
            (run / "environment.json").write_text(json.dumps({"artifact_completion_sha256": "hash",
                "model": {"revision": "revision"}}))
            for old_lock in ("previous-lock", None):
                with self.subTest(old_lock=old_lock):
                    (run / "request.json").write_text(json.dumps({
                        "kind": "smoke", "dependency_lock_sha256": old_lock}))
                    with self.assertRaises(ValueError):
                        control.successful_pretrained_smoke(root, "hash", "revision", "current-lock")
            (run / "request.json").write_text(json.dumps({
                "kind": "smoke", "dependency_lock_sha256": "current-lock"}))
            self.assertEqual(control.successful_pretrained_smoke(root, "hash", "revision", "current-lock"),
                             "candidate")

    def test_zombie_worker_is_not_running(self):
        with patch.object(Path, "read_text", return_value="100 (python3) Z " + "0 " * 30):
            self.assertIsNone(control.process_birth(100))

    @unittest.skipIf(sys.platform == "win32", "fcntl lock semantics are exercised on Linux")
    def test_conflicting_experiment_cannot_enter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with control.execution_guard(root, "smoke"):
                with self.assertRaises(BlockingIOError):
                    with control.execution_guard(root, "pilot"):
                        self.fail("Competing experiment entered")
                with self.assertRaises(BlockingIOError):
                    with control.execution_guard(root, "setup"):
                        self.fail("Competing setup entered")
            with control.execution_guard(root, "pilot"):
                pass


if __name__ == "__main__":
    unittest.main()
