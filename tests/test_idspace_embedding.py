import unittest
from unittest.mock import patch

import torch
from torch import nn

from anytrain.module.idspace import Embedding, Layout


class EmbeddingTest(unittest.TestCase):
    def test_forward_routes_ids_by_block(self):
        layout = Layout(text=(0, 2), audio=(10, 12))
        text = nn.Embedding(2, 2)
        audio = nn.Embedding(2, 2)
        with torch.no_grad():
            text.weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
            audio.weight.copy_(torch.tensor([[0.0, 3.0], [0.0, 4.0]]))
        embed = Embedding(layout, text=text, audio=audio)

        output = embed(torch.tensor([[0, 10, 1, 11]]))

        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[[1.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 4.0]]]),
            )
        )
        self.assertIs(embed.embeddings["text"], text)
        self.assertIs(embed.embeddings["audio"], audio)
        self.assertEqual(embed.num_embeddings, 12)
        self.assertEqual(embed.vocab_size, 12)

    def test_homogeneous_inference_allows_unselected_blocks(self):
        layout = Layout(text=(0, 2), audio=(10, 12))
        text = nn.Embedding(2, 2)
        audio = nn.Embedding(2, 2)
        embed = Embedding(layout, text=text, audio=audio)

        with torch.no_grad():
            output = embed(torch.tensor([0, 1]))

        self.assertTrue(torch.equal(output, text(torch.tensor([0, 1]))))

    def test_training_keeps_unselected_block_gradient_absent(self):
        layout = Layout(text=(0, 2), audio=(10, 12))
        text = nn.Embedding(2, 2)
        audio = nn.Embedding(2, 2)
        embed = Embedding(layout, text=text, audio=audio)
        optimizer = torch.optim.AdamW(embed.parameters(), lr=0.1, weight_decay=0.1)
        audio_before = audio.weight.detach().clone()

        with patch.object(
            torch.Tensor,
            "__bool__",
            side_effect=AssertionError("embedding routing must not test device tensors in Python"),
        ):
            output = embed(torch.tensor([0, 1]))
        output.sum().backward()

        self.assertIsNotNone(text.weight.grad)
        self.assertIsNone(audio.weight.grad)
        optimizer.step()
        self.assertTrue(torch.equal(audio.weight, audio_before))

    def test_adapter_path_keeps_unselected_parameters_out_of_autograd(self):
        layout = Layout(text=(0, 2), audio=(2, 4))
        text = nn.Embedding(2, 2)
        audio = nn.Embedding(2, 2)
        audio_adapter = nn.Linear(2, 2, bias=False)
        embed = Embedding(
            layout,
            adapters={"audio": audio_adapter},
            text=text,
            audio=audio,
        )

        with patch.object(
            torch.Tensor,
            "__bool__",
            side_effect=AssertionError("embedding routing must not test device tensors in Python"),
        ):
            output = embed(torch.tensor([0, 1]))
        output.sum().backward()

        self.assertIsNotNone(text.weight.grad)
        self.assertIsNone(audio.weight.grad)
        self.assertIsNone(audio_adapter.weight.grad)

    def test_trusted_selected_blocks_skips_device_routing_summary(self):
        layout = Layout(text=(0, 2), audio=(2, 4))
        text = nn.Embedding(2, 2)
        audio = nn.Embedding(2, 2)
        adapter = nn.Linear(2, 2, bias=False)
        embed = Embedding(layout, adapters={"audio": adapter}, text=text, audio=audio)

        with patch(
            "anytrain.module.idspace.embedding._selected_blocks",
            side_effect=AssertionError("trusted routing must not rescan device ids"),
        ):
            output = embed(
                torch.tensor([0, 1]),
                selected_blocks=frozenset({"text"}),
                validate=False,
            )
        output.sum().backward()

        self.assertIsNotNone(text.weight.grad)
        self.assertIsNone(audio.weight.grad)
        self.assertIsNone(adapter.weight.grad)

    def test_trusted_selected_blocks_bypasses_homogeneous_validation_path(self):
        layout = Layout(text=(0, 2), audio=(2, 4))
        text = nn.Embedding(2, 2)
        audio = nn.Embedding(2, 2)
        embed = Embedding(layout, text=text, audio=audio)

        with (
            patch(
                "anytrain.module.idspace.embedding._selected_blocks",
                side_effect=AssertionError("trusted routing must not rescan device ids"),
            ),
            patch(
                "anytrain.module.idspace.embedding._validate_covered",
                side_effect=AssertionError("trusted routing must not synchronize coverage"),
            ),
        ):
            output = embed(
                torch.tensor([0, 1]),
                selected_blocks={"text"},
                validate=False,
            )
        output.sum().backward()

        self.assertIsNotNone(text.weight.grad)
        self.assertIsNone(audio.weight.grad)

    def test_trusted_selected_blocks_requires_known_nonempty_names(self):
        layout = Layout(text=(0, 2), audio=(2, 4))
        embed = Embedding(layout, text=nn.Embedding(2, 2), audio=nn.Embedding(2, 2))

        with self.assertRaisesRegex(ValueError, "required"):
            embed(torch.tensor([0]), validate=False)
        with self.assertRaisesRegex(ValueError, "unknown blocks"):
            embed(torch.tensor([0]), selected_blocks={"image"}, validate=False)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            embed(torch.tensor([0]), selected_blocks=set(), validate=False)

    def test_homogeneous_inference_falls_back_for_mixed_dtypes(self):
        layout = Layout(text=(0, 2), audio=(10, 12))
        text = nn.Embedding(2, 2, dtype=torch.float32)
        audio = nn.Embedding(2, 2, dtype=torch.float64)
        embed = Embedding(layout, text=text, audio=audio)

        with torch.no_grad():
            output = embed(torch.tensor([0, 1]))

        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, text(torch.tensor([0, 1]))))

    def test_forward_applies_block_adapter(self):
        layout = Layout(text=(0, 2), audio=(2, 4))
        text = nn.Embedding(2, 4)
        audio = nn.Embedding(2, 2)
        adapter = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            text.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            audio.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            adapter.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0]]))
        embed = Embedding(layout, adapters={"audio": adapter}, text=text, audio=audio)

        output = embed(torch.tensor([0, 2, 3]))
        expected = torch.stack([text.weight[0], adapter(audio.weight[0]), adapter(audio.weight[1])])

        self.assertTrue(torch.equal(output, expected))

    def test_forward_keeps_gradients(self):
        layout = Layout(text=(0, 2), audio=(2, 3))
        embed = Embedding(layout, text=nn.Embedding(2, 2), audio=nn.Embedding(1, 2))

        loss = embed(torch.tensor([0, 1, 2])).sum()
        loss.backward()

        self.assertIsNotNone(embed.embeddings["text"].weight.grad)
        self.assertIsNotNone(embed.embeddings["audio"].weight.grad)

    def test_forward_rejects_empty_input(self):
        layout = Layout(text=(0, 2), audio=(2, 3))
        embed = Embedding(layout, text=nn.Embedding(2, 4), audio=nn.Embedding(1, 4))

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            embed(torch.empty(2, 0, dtype=torch.long))

    def test_forward_rejects_invalid_ids(self):
        layout = Layout(text=(0, 2))
        embed = Embedding(layout, text=nn.Embedding(2, 2))

        with (
            patch.object(
                torch.Tensor,
                "__bool__",
                side_effect=AssertionError(
                    "embedding routing must not test device tensors in Python"
                ),
            ),
            self.assertRaisesRegex(ValueError, "outside space"),
        ):
            embed(torch.tensor([2]))

    def test_forward_rejects_mixed_output_dims(self):
        layout = Layout(text=(0, 1), audio=(1, 2))
        with self.assertRaisesRegex(ValueError, "same output dim"):
            embed = Embedding(layout, text=nn.Embedding(1, 2), audio=nn.Embedding(1, 3))
            embed(torch.tensor([0, 1]))

    def test_init_requires_all_and_only_layout_blocks(self):
        layout = Layout(text=(0, 2), audio=(2, 3))

        with self.assertRaisesRegex(ValueError, "missing embeddings"):
            Embedding(layout, text=nn.Embedding(2, 2))
        with self.assertRaisesRegex(ValueError, "unknown embedding"):
            Embedding(
                layout,
                text=nn.Embedding(2, 2),
                audio=nn.Embedding(1, 2),
                image=nn.Embedding(1, 2),
            )
        with self.assertRaisesRegex(ValueError, "unknown adapter"):
            Embedding(
                layout,
                adapters={"image": nn.Identity()},
                text=nn.Embedding(2, 2),
                audio=nn.Embedding(1, 2),
            )

    def test_init_rejects_mismatched_embedding_size(self):
        layout = Layout(text=(0, 2))

        with self.assertRaisesRegex(ValueError, "block size"):
            Embedding(layout, text=nn.Embedding(3, 2))


if __name__ == "__main__":
    unittest.main()
