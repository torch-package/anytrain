import unittest

import torch

from anytrain.module.idspace import Layout


class LayoutTest(unittest.TestCase):
    def test_layout_defines_named_non_overlapping_blocks(self):
        layout = Layout(text=(0, 4), audio=(10, 13))

        self.assertEqual(layout.blocks, {"text": (0, 4), "audio": (10, 13)})
        self.assertEqual(layout.block_names, ("text", "audio"))
        self.assertEqual(layout.vocab_size, 13)
        self.assertEqual(layout.block("text"), (0, 4))
        self.assertEqual(layout.block_name_for_id(0), "text")
        self.assertEqual(layout.block_name_for_id(12), "audio")
        with self.assertRaises(TypeError):
            layout.blocks["image"] = (20, 21)

    def test_to_global_converts_local_ids(self):
        layout = Layout(text=(3, 7), audio=(7, 9))

        ids = torch.tensor([[0, 1], [2, 3]])
        global_ids = layout.to_global("text", ids)

        self.assertTrue(torch.equal(global_ids, torch.tensor([[3, 4], [5, 6]])))
        self.assertEqual(global_ids.shape, ids.shape)
        self.assertEqual(global_ids.device, ids.device)
        self.assertEqual(global_ids.dtype, ids.dtype)

    def test_to_local_converts_ids_from_one_block(self):
        layout = Layout(text=(3, 7), audio=(7, 9))

        ids = torch.tensor([[3, 4], [5, 6]])
        local_ids = layout.to_local(ids)

        self.assertTrue(torch.equal(local_ids, torch.tensor([[0, 1], [2, 3]])))
        self.assertEqual(local_ids.shape, ids.shape)
        self.assertEqual(local_ids.device, ids.device)
        self.assertEqual(local_ids.dtype, ids.dtype)

    def test_to_local_keeps_ignore_id(self):
        layout = Layout(text=(3, 7), audio=(7, 9))

        ids = torch.tensor([[3, -100], [5, 6]])
        local_ids = layout.to_local(ids, ignore=-100)

        self.assertTrue(torch.equal(local_ids, torch.tensor([[0, -100], [2, 3]])))
        self.assertEqual(local_ids.shape, ids.shape)
        self.assertEqual(local_ids.device, ids.device)
        self.assertEqual(local_ids.dtype, ids.dtype)

    def test_to_local_returns_all_ignore_ids_unchanged(self):
        layout = Layout(text=(3, 7))

        ids = torch.tensor([-100, -100])
        local_ids = layout.to_local(ids, ignore=-100)

        self.assertTrue(torch.equal(local_ids, ids))
        self.assertIsNot(local_ids, ids)

    def test_to_local_rejects_mixed_or_unknown_blocks(self):
        layout = Layout(text=(3, 7), audio=(7, 9))

        with self.assertRaisesRegex(ValueError, "same id block"):
            layout.to_local(torch.tensor([3, 7]))
        with self.assertRaisesRegex(ValueError, "outside all id blocks"):
            layout.to_local(torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "same id block"):
            layout.to_local(torch.tensor([3, -100, 7]), ignore=-100)

    def test_rejects_ids_outside_block(self):
        layout = Layout(text=(3, 7))

        with self.assertRaisesRegex(ValueError, "outside block"):
            layout.to_global("text", torch.tensor([-1]))

    def test_rejects_invalid_blocks(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            Layout()
        with self.assertRaisesRegex(TypeError, "tuple"):
            Layout(text=[0, 1])
        with self.assertRaisesRegex(ValueError, "greater than start"):
            Layout(text=(1, 1))
        with self.assertRaisesRegex(ValueError, "overlap"):
            Layout(text=(0, 4), audio=(3, 6))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            Layout(**{"": (0, 1)})
        with self.assertRaisesRegex(ValueError, "must not contain"):
            Layout(**{"text.main": (0, 1)})


if __name__ == "__main__":
    unittest.main()
