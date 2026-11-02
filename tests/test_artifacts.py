"""Persistent completion pointers and interrupted conversion recovery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from moe_shared.artifacts import atomic_json, contained_file, file_hash, publish_completion, resolve_artifacts, verify_model
from moe_shared.prepare import converter_hash, verified_progress_bytes


class ArtifactManifestTests(unittest.TestCase):
    def test_partial_download_capacity_is_credited_only_for_exact_etag(self):
        script = Path(__file__).resolve().parents[1] / "scripts/acquire_model.py"
        spec = importlib.util.spec_from_file_location("artifact_acquire_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = SimpleNamespace(rfilename="weights.safetensors", size=100,
                                   lfs={"sha256": "a" * 64}, blob_id="b" * 40)
            partial = root / ".cache/huggingface/download" / (item.rfilename + "." + "a" * 64 + ".incomplete")
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"x" * 60)
            self.assertEqual(module.resumable_download_bytes(root, item), 60)
            item.lfs = {"sha256": "c" * 64}
            self.assertEqual(module.resumable_download_bytes(root, item), 0)
            item.lfs = {"sha256": "a" * 64}
            (root / item.rfilename).write_bytes(b"invalid target")
            self.assertEqual(module.resumable_download_bytes(root, item), 0)

    def test_verified_republication_preserves_exact_completion_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "complete.json"
            first = {"complete": True, "revision": "abc", "backing_manifest_sha256": "123",
                     "completed_at_utc": "first time", "artifact_guard": {"free": 42}}
            actual, reused = publish_completion(path, first)
            self.assertFalse(reused)
            original, mtime = path.read_bytes(), path.stat().st_mtime_ns
            new_observation = dict(first, completed_at_utc="later time", artifact_guard={"free": 41})
            actual, reused = publish_completion(path, new_observation)
            self.assertTrue(reused)
            self.assertEqual(actual, first)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            _, reused = publish_completion(path, dict(new_observation, backing_manifest_sha256="456"))
            self.assertFalse(reused)

    def test_checkpoint_checksum_and_traversal_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_bytes(b"{}")
            saved = {"files": {"config.json": {"bytes": 2, "sha256": file_hash(root / "config.json")}}}
            self.assertIs(verify_model(root, saved), saved)
            (root / "config.json").write_bytes(b"[]")
            with self.assertRaisesRegex(IOError, "verification failed"):
                verify_model(root, saved)
            with self.assertRaises(ValueError):
                contained_file(root, "../elsewhere")

    def test_complete_pointer_resolves_exact_metadata_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, backing = root / "revision/model", root / "revision/backing"
            model.mkdir(parents=True)
            backing.mkdir(parents=True)
            (model / "config.json").write_bytes(b"{}")
            provenance = {"model_id": "fixture", "revision": "a" * 40,
                          "files": {"config.json": {"bytes": 2, "sha256": file_hash(model / "config.json")}}}
            atomic_json(model / "model-provenance.json", provenance)
            manifest = {"converter_source_sha256": converter_hash()}
            atomic_json(backing / "manifest.json", manifest)
            pointer = {"schema_version": 1, "complete": True,
                       "model_dir": "revision/model", "backing_dir": "revision/backing",
                       "model_id": "fixture", "revision": "a" * 40,
                       "converter_source_sha256": converter_hash(),
                       "model_provenance_sha256": file_hash(model / "model-provenance.json"),
                       "backing_manifest_sha256": file_hash(backing / "manifest.json")}
            atomic_json(root / "complete.json", pointer)
            resolved_model, resolved_backing, actual = resolve_artifacts(root, verify=False)
            self.assertEqual((resolved_model, resolved_backing), (model, backing))
            self.assertEqual(actual, pointer)
            (backing / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(IOError, "metadata changed"):
                resolve_artifacts(root, verify=False)

    def test_progress_capacity_counts_verified_completed_files_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            backing = Path(temporary) / "backing"
            staging = backing.with_name("backing.preparing")
            (staging / ".receipts").mkdir(parents=True)
            atomic_json(staging / ".progress.json", {})
            (staging / "good.bin").write_bytes(b"abcd")
            (staging / "bad.bin").write_bytes(b"wrong")
            atomic_json(staging / ".receipts/good.json", {"file": "good.bin", "nbytes": 4,
                         "sha256": hashlib.sha256(b"abcd").hexdigest()})
            atomic_json(staging / ".receipts/bad.json", {"file": "bad.bin", "nbytes": 5,
                         "sha256": hashlib.sha256(b"right").hexdigest()})
            (staging / "unfinished.bin.partial").write_bytes(b"x" * 128)
            self.assertEqual(verified_progress_bytes(backing), 4)


try:
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch required")
class ResumableConversionTests(unittest.TestCase):
    def test_interrupted_conversion_reuses_completed_bytes_and_never_publishes_early(self):
        from moe_shared.prepare import prepare_model, verify_backing
        with tempfile.TemporaryDirectory(prefix="moe-artifact-recovery-fixture-") as temporary:
            root = Path(temporary)
            config = OlmoeConfig(vocab_size=16, hidden_size=8, intermediate_size=8,
                                num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=2, num_experts=2, num_experts_per_tok=1)
            model = OlmoeForCausalLM(config)
            model.save_pretrained(root / "model", safe_serialization=True)
            del model
            completed = []

            def interrupt(path):
                completed.append((path, path.stat().st_mtime_ns, file_hash(path)))
                raise InterruptedError("intentional conversion interruption")

            with self.assertRaises(InterruptedError):
                prepare_model(root / "model", root / "backing", _after_file=interrupt)
            self.assertFalse((root / "backing/manifest.json").exists())
            self.assertEqual(len(completed), 1)
            manifest = prepare_model(root / "model", root / "backing")
            self.assertEqual(manifest["recovery"]["reused_verified_files"], 1)
            original_path, mtime, digest = completed[0]
            final_path = root / "backing" / original_path.relative_to(root / "backing.preparing")
            self.assertEqual(final_path.stat().st_mtime_ns, mtime)
            self.assertEqual(file_hash(final_path), digest)
            verify_backing(root / "backing")
            self.assertEqual(prepare_model(root / "model", root / "backing"), manifest)


if __name__ == "__main__":
    unittest.main()
