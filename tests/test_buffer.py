import unittest

import torch
from torch import nn

from anytrain._buffer import register_buffer


class RegisterBufferTest(unittest.TestCase):
    def _assert_registered_buffer_semantics(self, *, modern: bool) -> None:
        module = nn.Module()
        persistent = torch.tensor([1.0, 2.0])
        transient = torch.tensor([3.0, 4.0])

        self.assertIsNone(register_buffer(module, "persistent", persistent))
        self.assertIsNone(register_buffer(module, "transient", transient, persistent=False))

        self.assertTrue(torch.equal(module.get_buffer("persistent"), persistent))
        self.assertTrue(torch.equal(module.get_buffer("transient"), transient))
        if modern:
            self.assertIsInstance(module.persistent, nn.Buffer)
            self.assertIsInstance(module.transient, nn.Buffer)
        else:
            self.assertIs(type(module.persistent), torch.Tensor)
            self.assertIs(type(module.transient), torch.Tensor)

        named_buffers = dict(module.named_buffers())
        self.assertEqual(set(named_buffers), {"persistent", "transient"})
        self.assertTrue(torch.equal(named_buffers["persistent"], persistent))
        self.assertTrue(torch.equal(named_buffers["transient"], transient))

        state = module.state_dict()
        self.assertEqual(set(state), {"persistent"})
        self.assertTrue(torch.equal(state["persistent"], persistent))

        module.to(dtype=torch.float64)
        self.assertEqual(module.persistent.dtype, torch.float64)
        self.assertEqual(module.transient.dtype, torch.float64)

    @unittest.skipUnless(
        hasattr(nn, "Buffer"),
        "requires torch.nn.Buffer",
    )
    def test_modern_buffer_path(self):
        self._assert_registered_buffer_semantics(modern=True)

    def test_fallback_path_without_nn_buffer(self):
        had_buffer = hasattr(nn, "Buffer")
        original_buffer = getattr(nn, "Buffer", None)
        if had_buffer:
            delattr(nn, "Buffer")
        try:
            self._assert_registered_buffer_semantics(modern=False)
        finally:
            if had_buffer:
                nn.Buffer = original_buffer


if __name__ == "__main__":
    unittest.main()
