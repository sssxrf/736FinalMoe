import unittest
from moe_shared.workload import ArrivalQueue, distribution, make_trajectories


class Words:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


class WorkloadTests(unittest.TestCase):
    def test_later_turn_is_not_admitted_before_completion_plus_think(self):
        q = ArrivalQueue()
        q.schedule("a", 0, 10)
        q.schedule("b", 0, 10)
        self.assertEqual(len(q.admit(10)), 2)
        self.assertEqual(q.next(), ("a", 0))
        # A completes at 12; B gets another step while A is paused.
        q.schedule("a", 1, 12 + 2)
        self.assertEqual(q.admit(13.9), [])
        self.assertEqual(q.next(), ("b", 0))
        self.assertEqual(q.admit(14), [("a", 1, 14)])

    def test_actual_token_ids_reused_and_segments_contiguous(self):
        cfg = dict(paused_prefix_tokens=8, active_prefix_tokens=7, first_turn_tokens=2,
                   active_output_tokens=5, return_prompt_tokens=3, return_output_tokens=2,
                   think_seconds=.1)
        a = make_trajectories(Words(), cfg)
        b = make_trajectories(Words(), cfg)
        self.assertEqual(a, b)
        turns = a["sessions"][0]["turns"]
        self.assertEqual(sum([t["prompt"] + t["output"] for t in turns], []), list(range(15)))

    def test_summary_has_sample_counts_not_extreme_quantiles(self):
        self.assertEqual(distribution([3, 1])["median"], 2)
        self.assertEqual(distribution([])["count"], 0)
        self.assertNotIn("p99", distribution([1]))


if __name__ == "__main__":
    unittest.main()
