"""Small mechanism fixture; never substitutes for the pretrained smoke run."""

import gc
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache, OlmoeConfig, OlmoeForCausalLM
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA PyTorch required")
class TinyGPUIntegrationTest(unittest.TestCase):
    @staticmethod
    def create_checkpoint(root):
        torch.manual_seed(17)
        config = OlmoeConfig(vocab_size=64, hidden_size=32, intermediate_size=24,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=4, num_experts=4,
                            num_experts_per_tok=2, max_position_embeddings=128,
                            tie_word_embeddings=False)
        config._attn_implementation = "eager"
        source = OlmoeForCausalLM(config)
        source.save_pretrained(root / "model", safe_serialization=True)
        del source

    @staticmethod
    def load_resident(root):
        # Match the real experiment's load path. A blanket .to(dtype=BF16)
        # would also round nonpersistent RoPE buffers, which are regenerated
        # in float32 when a checkpoint is loaded and are absent from its files.
        return AutoModelForCausalLM.from_pretrained(
            root / "model", torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            attn_implementation="eager", local_files_only=True,
            trust_remote_code=False).eval()

    def test_real_tensor_misses_and_all_kv_actions(self):
        from moe_shared.experts import load_offloaded
        from moe_shared.kv import KVManager
        from moe_shared.pool import SharedPool
        from moe_shared.prepare import prepare_model

        with tempfile.TemporaryDirectory(prefix="moe-shared-fixture-") as tmp:
            root = Path(tmp)
            self.create_checkpoint(root)
            reference = self.load_resident(root)
            expected_parameters = {name: value.detach().cpu().clone()
                                   for name, value in reference.named_parameters()}
            expected_buffers = {name: value.detach().cpu().clone()
                                for name, value in reference.named_buffers()}
            self.assertTrue(expected_buffers)
            self.assertTrue(all(value.dtype == torch.float32 for value in expected_buffers.values()))
            prefix = torch.tensor([[1, 7, 9, 3]], device="cuda")
            next_token = torch.tensor([[11]], device="cuda")
            with torch.inference_mode():
                original = DynamicCache()
                result = reference(prefix, past_key_values=original, use_cache=True)
                expected_prefill = result.logits.cpu()
                result = reference(next_token, past_key_values=original, use_cache=True)
                expected_next = result.logits.cpu()
            del reference, result, original
            gc.collect()
            torch.cuda.empty_cache()
            manifest = prepare_model(root / "model", root / "backing")
            # A single GPU slot forces eviction even within each layer's top-2.
            for action in ("retain", "spill", "recompute"):
                with self.subTest(action=action):
                    pool = SharedPool(128 * 1024)
                    fixture_bytes = sum(t.numel() * t.element_size()
                                        for t in [*expected_parameters.values(), *expected_buffers.values(),
                                                  expected_prefill, expected_next])
                    pool.reserve("fixture_reference", fixture_bytes, "validation")
                    model, experts = load_offloaded(root / "model", root / "backing", pool,
                                                   manifest["max_expert_bytes"])
                    self.assertEqual(set(dict(model.named_buffers())), set(expected_buffers))
                    for name, value in model.named_buffers():
                        torch.testing.assert_close(value.cpu(), expected_buffers[name], rtol=0, atol=0)
                    for name, value in model.named_parameters():
                        torch.testing.assert_close(value.cpu(), expected_parameters[name], rtol=0, atol=0)
                    # Check actual materialized expert weights as well as dense
                    # parameters and nonpersistent buffers before testing logits.
                    for key in manifest["experts"]:
                        layer, expert = key.split(".")
                        expert_module = experts.get(key)
                        for name, value in expert_module.named_parameters():
                            full_name = f"model.layers.{layer}.mlp.experts.{expert}.{name}"
                            torch.testing.assert_close(value.cpu(), expected_parameters[full_name], rtol=0, atol=0)
                    del expert_module, value
                    experts.close()  # inference starts with cold application caches
                    manager = KVManager(pool, root / "backing" / "kv" / action)
                    with torch.inference_mode():
                        cache = DynamicCache()
                        result = model(prefix, past_key_values=cache, use_cache=True)
                        torch.testing.assert_close(result.logits.cpu(), expected_prefill, rtol=0, atol=0)
                        result = None
                        handle = manager.pause(cache, action, "fixture-session")
                        self.assertEqual(cache.get_seq_length(), 0)
                        self.assertEqual(len(cache.key_cache), 0)
                        self.assertGreater(handle.nbytes, 0)
                        if action == "retain":
                            self.assertEqual(pool.stats()["by_category"]["kv"], handle.nbytes)
                            self.assertTrue(all(t.device.type == "cpu" for t in handle.tensors))
                        if action == "spill":
                            self.assertFalse(handle.tensors)
                            self.assertTrue(handle.path.is_file())
                            self.assertNotIn(handle.tag, pool.entries)
                        # Another active session executes while this KV is paused.
                        active = model(prefix[:, :2], past_key_values=DynamicCache(), use_cache=True)
                        del active
                        cache = manager.resume(handle)
                        if cache is None:
                            cache = DynamicCache()
                            replay = model(prefix, past_key_values=cache, use_cache=True)
                            del replay
                        result = model(next_token, past_key_values=cache, use_cache=True)
                        torch.testing.assert_close(result.logits.cpu(), expected_next, rtol=0, atol=0)
                        self.assertTrue(handle.consumed)
                        with self.assertRaises(ValueError):
                            manager.resume(handle)
                        del result, cache
                    self.assertGreater(experts.stats()["host_misses"], 0)
                    self.assertGreater(experts.stats()["gpu_evictions"], 0)
                    self.assertGreater(experts.stats()["cuda_executions"], 0)
                    self.assertLessEqual(experts.stats()["gpu_peak_bytes"], manifest["max_expert_bytes"])
                    self.assertLessEqual(pool.stats()["peak_bytes"], pool.budget_bytes)
                    manager.close()
                    experts.close()
                    pool.release("fixture_reference")
                    self.assertEqual(pool.used_bytes, 0)
                    del model, experts, manager
                    gc.collect()
                    torch.cuda.empty_cache()

    def test_token_scheduler_and_returning_turns(self):
        from moe_shared.experts import load_offloaded
        from moe_shared.kv import KVManager
        from moe_shared.pool import SharedPool
        from moe_shared.prepare import prepare_model
        from moe_shared.run import Recorder, execute_schedule

        trajectories = {"sessions": [
            {"id": "paused", "turns": [
                {"prompt": [1, 7, 9, 3], "output": [11, 13], "think_seconds": 0.0},
                {"prompt": [5, 2], "output": [17, 19], "think_seconds": 0.002}]},
            {"id": "active", "turns": [
                {"prompt": [2, 4, 6], "output": [8, 10, 12, 14, 16], "think_seconds": 0.0}]},
        ]}
        config = {"max_context_tokens": 32, "total_gpu_budget_bytes": 256 * 1024 * 1024,
                  "logit_atol": 0.004, "logit_rtol": 0.01, "minimum_argmax_agreement": 1.0}
        # The recompute path uses a different GEMM batch shape from incremental
        # decoding, so this scheduler check has an explicit BF16 tolerance. The
        # direct cache-transition test above still requires exact equality.
        with tempfile.TemporaryDirectory(prefix="moe-scheduler-fixture-") as tmp:
            root = Path(tmp)
            self.create_checkpoint(root)
            event = Recorder(root)
            try:
                event.case = "fixture-resident"
                reference = self.load_resident(root)
                references, _, baseline, reference_summary = execute_schedule(
                    reference, trajectories, config, event, "resident")
                self.assertEqual(len(references), 9)
                self.assertGreater(reference_summary["active_steps_while_paused"], 0)
                del reference
                gc.collect()
                torch.cuda.empty_cache()
                manifest = prepare_model(root / "model", root / "backing")
                baseline_tokens = {(m["session"], m["turn"], m["token_index"]): m["forced_token_id"]
                                   for m in baseline}
                for action in ("retain", "spill", "recompute"):
                    with self.subTest(action=action):
                        event.case = "fixture-" + action
                        pool = SharedPool(128 * 1024)
                        validation_bytes = sum(t.numel()*t.element_size() for t in references.values())
                        validation_bytes += 6 * 64 * 4
                        pool.reserve("fixture_reference", validation_bytes, "validation")
                        model, experts = load_offloaded(root / "model", root / "backing", pool,
                                                       manifest["max_expert_bytes"], event)
                        manager = KVManager(pool, root / "backing" / "kv" / action, event)
                        _, checks, measures, summary = execute_schedule(
                            model, trajectories, config, event, action, references, manager)
                        self.assertEqual(len(checks), 9)
                        self.assertEqual(sum(c["returning_turn"] for c in checks), 2)
                        self.assertTrue(summary["correctness"]["passed"], summary["correctness"])
                        self.assertGreater(summary["active_steps_while_paused"], 0)
                        self.assertEqual(summary["initial_ttft_seconds"]["count"], 2)
                        self.assertEqual(summary["returning_ttft_seconds"]["count"], 1)
                        self.assertEqual({(m["session"], m["turn"], m["token_index"]): m["forced_token_id"]
                                          for m in measures}, baseline_tokens)
                        self.assertGreater(experts.stats()["host_misses"], 0)
                        self.assertGreater(experts.stats()["cuda_executions"], 0)
                        self.assertEqual(manager.stats()[action], 1)
                        self.assertEqual(manager.stats()["live_handles"], 0)
                        manager.close()
                        experts.close()
                        pool.release("fixture_reference")
                        self.assertEqual(pool.used_bytes, 0)
                        del model, experts, manager
                        gc.collect()
                        torch.cuda.empty_cache()
            finally:
                event.close()
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertTrue({"arrival", "queue", "pause", "resume", "completion", "token"}
                            .issubset({entry["event"] for entry in events}))
            self.assertTrue(all(entry["stale_gpu_cache_entries"] == 0
                                for entry in events if entry["event"] == "kv_pause"))


@unittest.skipUnless(torch is not None, "PyTorch required")
class KVIntegrityTest(unittest.TestCase):
    def test_corrupted_spill_rejected_and_reservation_released(self):
        from moe_shared.kv import KVManager
        from moe_shared.pool import SharedPool

        with tempfile.TemporaryDirectory(prefix="moe-kv-fixture-") as tmp:
            pool = SharedPool(4096)
            manager = KVManager(pool, tmp, device="cpu")
            cache = DynamicCache()
            cache.update(torch.ones((1, 2, 3, 4), dtype=torch.bfloat16),
                         torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16), 0)
            handle = manager.pause(cache, "spill", "fixture")
            self.assertEqual(pool.used_bytes, 0)
            with handle.path.open("r+b") as stream:
                stream.write(b"\x00\x00")
            with self.assertRaisesRegex(IOError, "checksum"):
                manager.resume(handle)
            self.assertEqual(pool.used_bytes, 0)
            manager.close()


if __name__ == "__main__":
    unittest.main()
