import unittest

from moe_shared.pool import BudgetExceeded, SharedPool


class PoolTest(unittest.TestCase):
    def test_retained_kv_evicts_actual_oldest_expert(self):
        pool = SharedPool(100)
        tensors = {"e0": object(), "e1": object()}
        for tag in tensors.copy():
            pool.reserve(tag, 40, "expert", evictable=True,
                         on_evict=lambda tag=tag: tensors.pop(tag))
        pool.touch("e0")
        pool.reserve("kv", 50, "kv")
        self.assertEqual(set(tensors), {"e0"})
        self.assertEqual(pool.stats()["by_category"], {"expert": 40, "kv": 50})
        self.assertEqual(pool.stats()["evictions"], 1)
        self.assertLessEqual(pool.peak_bytes, 100)

    def test_failure_does_not_discard_existing_entries(self):
        pool = SharedPool(100)
        pool.reserve("kv", 70, "kv")
        evicted = []
        pool.reserve("e", 20, "expert", evictable=True,
                     on_evict=lambda: evicted.append("e"))
        with self.assertRaises(BudgetExceeded):
            pool.reserve("stage", 31, "staging")
        self.assertEqual(pool.used_bytes, 90)
        self.assertFalse(evicted)

    def test_reservations_and_validation(self):
        pool = SharedPool(100)
        pool.reserve("stage", 100, "staging")
        with self.assertRaises(BudgetExceeded):
            pool.reserve("expert", 1)
        with self.assertRaises(ValueError):
            pool.reserve("stage", 0)
        pool.release("stage")
        self.assertEqual(pool.used_bytes, 0)
        with self.assertRaises(ValueError):
            pool.reserve("bad", -1)
        with self.assertRaises(ValueError):
            pool.reserve("bad", 2, "kv", evictable=True, on_evict=lambda: None)


if __name__ == "__main__":
    unittest.main()
