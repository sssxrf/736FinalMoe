import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moe_shared.audit import check_space, directory_bytes, parse_key_values, parse_io_stat, cgroup_paths


class AuditTests(unittest.TestCase):
    def test_counters(self):
        self.assertEqual(parse_key_values("read_bytes: 123\nwrite_bytes: 456"),
                         {"read_bytes": 123, "write_bytes": 456})
        self.assertEqual(parse_key_values({"unavailable": "denied"}), {"unavailable": "denied"})
        self.assertEqual(parse_io_stat("253:0 rbytes=4096 wbytes=1024 rios=1\n259:2 rbytes=0"),
                         {"253:0": {"rbytes": 4096, "wbytes": 1024, "rios": 1}, "259:2": {"rbytes": 0}})

    def test_scratch_cap_and_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").write_bytes(b"abcd")
            self.assertEqual(directory_bytes(tmp), 4)
            with patch("moe_shared.audit.shutil.disk_usage") as usage:
                usage.return_value.free = 100
                self.assertEqual(check_space(tmp, 5, 10, 90)["task_logical_bytes"], 4)
                with self.assertRaisesRegex(RuntimeError, "cap"):
                    check_space(tmp, 7, 10, 0)
                with self.assertRaisesRegex(RuntimeError, "headroom"):
                    check_space(tmp, 5, 100, 96)
                with self.assertRaises(ValueError):
                    check_space(tmp, -1)

    def test_visible_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cg" / "child").mkdir(parents=True)
            (root / "proc").write_text("0::/child\n")
            self.assertEqual(cgroup_paths(root / "proc", root / "cg"),
                             [root / "cg" / "child", root / "cg"])


if __name__ == "__main__":
    unittest.main()
