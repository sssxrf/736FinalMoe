import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from moe_shared.lifecycle import SENTINEL, identity_checkpoint
from moe_shared.staging import OWNER_FILE, recovery_probe, remove_recovery_stage, stage_backing


class StagingRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        canonical = root / "canonical"
        (canonical / "experts").mkdir(parents=True)
        data = b"real byte payload for recovery test\x00\x01"
        (canonical / "experts" / "0.0.bin").write_bytes(data)
        manifest = {"format_version": 1, "experts": {"0.0": {
            "file": "experts/0.0.bin", "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "layout": []}}, "dense": {}}
        (canonical / "manifest.json").write_text(json.dumps(manifest))
        return canonical, data

    def test_rebuild_missing_disposable_stage_preserves_canonical(self):
        with tempfile.TemporaryDirectory(prefix="moe-stage-recovery-fixture-") as temporary:
            root = Path(temporary).resolve()
            canonical, data = self.fixture(root)
            stage = root / "disposable-recovery-stage"
            with patch("moe_shared.staging.check_space", return_value={"fixture": True}):
                stage_backing(canonical, stage, root / "first")
                canonical_hash = hashlib.sha256((canonical / "manifest.json").read_bytes()).hexdigest()
                # Only this dedicated fixture stage is removed. Verify the final
                # resolved path and ownership before the deliberately destructive test.
                self.assertEqual(stage.resolve(), root / "disposable-recovery-stage")
                self.assertTrue(stage.resolve().is_relative_to(root))
                self.assertNotEqual(stage.resolve(), canonical.resolve())
                self.assertTrue((stage / OWNER_FILE).is_file())
                shutil.rmtree(stage)
                self.assertFalse(stage.exists())
                self.assertEqual((canonical / "experts/0.0.bin").read_bytes(), data)
                stage_backing(canonical, stage, root / "second")
                self.assertEqual((stage / "experts/0.0.bin").read_bytes(), data)
                self.assertEqual(hashlib.sha256((canonical / "manifest.json").read_bytes()).hexdigest(), canonical_hash)
                stage_backing(canonical, stage, root / "third")
                report = json.loads((root / "third/staging.json").read_text())
                self.assertEqual(report["reused_files"], 1)
                self.assertEqual(report["copied_bytes"], 0)

    def test_corruption_is_repaired_without_touching_valid_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            canonical, data = self.fixture(root)
            stage = root / "stage"
            with patch("moe_shared.staging.check_space", return_value={}):
                stage_backing(canonical, stage, root / "first")
                (stage / "experts/0.0.bin").write_bytes(b"corrupted")
                stage_backing(canonical, stage, root / "second")
                self.assertEqual((stage / "experts/0.0.bin").read_bytes(), data)
                self.assertEqual((canonical / "experts/0.0.bin").read_bytes(), data)

    def test_stage_must_not_overlap_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            canonical, _ = self.fixture(root)
            for stage in (canonical, canonical / "stage", root):
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "overlap"):
                    stage_backing(canonical, stage, root / "logs")

    def test_sentinel_disappearance_is_observed_without_recreation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "target"
            first = identity_checkpoint(root / "logs", "before", {"artifact": target}, True)
            self.assertFalse(first["paths"]["artifact"]["sentinel"]["exists"])
            self.assertTrue(first["paths"]["artifact"]["after_sentinel_creation"]["sentinel"]["exists"])
            (target / SENTINEL).unlink()
            second = identity_checkpoint(root / "logs", "after", {"artifact": target})
            self.assertFalse(second["paths"]["artifact"]["sentinel"]["exists"])
            self.assertFalse((target / SENTINEL).exists())
            self.assertIn("boot ID alone", second["interpretation"])

    def test_recovery_probe_reports_deletion_reconstruction_and_source_integrity(self):
        with tempfile.TemporaryDirectory(prefix="moe-probe-fixture-") as temporary:
            root = Path(temporary).resolve()
            canonical, data = self.fixture(root)
            with patch("moe_shared.artifacts.resolve_artifacts", return_value=(root / "model", canonical, {"revision": "fixture"})), \
                 patch("moe_shared.staging.check_space", return_value={}):
                report = recovery_probe(root / "artifacts", root / "scratch", root / "reports" / "fixture-run")
            self.assertTrue(report["complete"])
            self.assertTrue(report["passed"])
            self.assertFalse(report["stage_exists_after_delete"])
            self.assertEqual(report["initial_stage_sha256"], report["reconstructed_stage_sha256"])
            self.assertTrue(report["canonical_unchanged_after_reconstruction"])
            self.assertEqual((canonical / "experts/0.0.bin").read_bytes(), data)
            with self.assertRaises(ValueError):
                remove_recovery_stage(canonical, root / "scratch", canonical)


if __name__ == "__main__":
    unittest.main()
