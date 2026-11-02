import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("snapshot", Path(__file__).resolve().parents[1] / "scripts/snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class SnapshotTests(unittest.TestCase):
    def test_includes_working_bytes_and_excludes_local_secret_and_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            (root / "configs").mkdir(parents=True)
            (root / "moe_shared").mkdir()
            (root / "results").mkdir()
            (root / "README.md").write_text("working tree, not HEAD")
            (root / "moe_shared/new.py").write_text("new = True\n")
            (root / "configs/machine.local.json").write_text('{"secret": "do not upload"}')
            (root / "configs/.env").write_text("PRIVATE=secret")
            (root / "results/measurement.json").write_text("{}")
            metadata = snapshot.create_snapshot(root, Path(temporary) / "out")
            files = snapshot.verified_members(Path(metadata["archive"]), metadata["archive_sha256"])
            self.assertEqual(set(files), {"README.md", "moe_shared/new.py", "source-manifest.json"})
            self.assertEqual(files["README.md"], b"working tree, not HEAD")

    def test_corrupted_transfer_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            archive = destination / "result.tar.gz"
            archive.write_bytes(snapshot.archive_bytes({"a.txt": b"original"}))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                snapshot.extract_results(archive, "0" * 64, destination / "out")
            self.assertFalse((destination / "out").exists())

    def test_manifest_integrity_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("../escape.txt", "/absolute", "a\\..\\escape"):
                raw = snapshot.archive_bytes({name: b"bad"})
                archive = root / "bad.tar.gz"
                archive.write_bytes(raw)
                with self.assertRaises(ValueError):
                    snapshot.verified_members(archive, snapshot.sha256(raw))
            data = b"measurement\n"
            manifest = {"files": {"events.jsonl": {"sha256": snapshot.sha256(data), "bytes": len(data)}}}
            raw = snapshot.archive_bytes({"events.jsonl": data, "fetch-manifest.json": json.dumps(manifest).encode()})
            archive = root / "good.tar.gz"
            archive.write_bytes(raw)
            receipt = snapshot.extract_results(archive, snapshot.sha256(raw), root / "out")
            self.assertEqual(receipt["verified_files"], 1)
            self.assertEqual((root / "out/events.jsonl").read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
