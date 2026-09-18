import copy
import unittest

import torch
from torch import nn

from anytrain._contract import canonical_sha256
from anytrain.module.embedding import EmbeddingBank, EmbeddingView


class EmbeddingTest(unittest.TestCase):
    def _view(self, *, prediction: bool = False) -> tuple[EmbeddingBank, EmbeddingView]:
        bank = EmbeddingBank(embedding_dim=2)
        shared = nn.Embedding(2, 2)
        output = nn.Embedding(2, 2)
        with torch.no_grad():
            shared.weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
            output.weight.copy_(torch.tensor([[0.0, 3.0], [0.0, 4.0]]))
        bank.add_owner("shared", shared, logical_names=("input",))
        bank.add_owner("output", output)
        prediction_indices = prediction_rows = None
        if prediction:
            prediction_indices = torch.tensor([1, 1, 0, 0])
            prediction_rows = torch.tensor([0, 1, 0, 1])
        view = EmbeddingView(
            bank,
            direction="output",
            num_embeddings=4,
            embedding_dim=2,
            stream_names=("semantic", "global"),
            stream_token_ranges={"semantic": ((0, 2),), "global": ((2, 4),)},
            stream_owner_names={"semantic": "shared", "global": "output"},
            owner_names=("shared", "output"),
            owner_indices=torch.tensor([0, 0, 1, 1]),
            owner_rows=torch.tensor([0, 1, 0, 1]),
            prediction_indices=prediction_indices,
            prediction_rows=prediction_rows,
            head_tied_by_stream={"semantic": not prediction, "global": True},
            contract_family="composite-v1",
            contract_grammar="audio-embedding-view-v1",
        )
        return bank, view

    def test_bank_registers_shared_owner_once(self):
        bank, view = self._view()
        self.assertIs(view.owner_parameter("semantic"), bank.owner("input"))
        self.assertIsNot(view.owner_parameter("semantic"), view.owner_parameter("global"))
        names = [name for name, _ in bank.named_parameters()]
        self.assertEqual(names, ["owners.shared.weight", "owners.output.weight"])
        self.assertEqual(bank.owner("input"), bank.owner("shared"))

    def test_view_routes_rows_and_prediction_rows(self):
        _, view = self._view(prediction=True)
        expected = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 4.0]]])
        torch.testing.assert_close(view(torch.tensor([[0, 1, 2, 3]])), expected)
        prediction = view.prediction_rows()
        torch.testing.assert_close(
            prediction,
            torch.tensor([[0.0, 3.0], [0.0, 4.0], [1.0, 0.0], [2.0, 0.0]]),
        )

    def test_untied_prediction_gradient_is_separate(self):
        bank, view = self._view(prediction=True)
        hidden = torch.ones(1, 2)
        view.logits(hidden, torch.tensor([0])).sum().backward()
        self.assertIsNone(bank.owner("shared").weight.grad)
        self.assertIsNotNone(bank.owner("output").weight.grad)

    def test_standalone_view_conversion_moves_canonical_bank(self):
        _, view = self._view()
        copied = copy.deepcopy(view)
        copied.double()
        self.assertIs(next(copied.bank.parameters()).dtype, torch.float64)
        self.assertIs(copied.bank, copied.owner)

    def test_owner_provider_avoids_structural_probing_and_keeps_metadata_overrides(self):
        class Owner(nn.Module):
            revision = 1

            @property
            def weight(self):
                raise AssertionError("contract must not materialize dense weights")

            @property
            def fsq_levels(self):
                raise AssertionError("explicit provider must bypass legacy inference")

            def embedding_contract_state(self):
                return {"implementation": "custom-v1", "revision": self.revision, "rows": 8}

        owner = Owner()
        bank = EmbeddingBank(2)
        bank.add_owner("custom", owner, state={"rows": 16})
        before = bank.contract_state()
        owners = before["owners"]
        assert isinstance(owners, dict)
        self.assertEqual(
            owners["custom"],
            {
                "implementation": "custom-v1",
                "revision": 1,
                "rows": 16,
            },
        )
        owner.revision = 2
        after = bank.contract_state()
        self.assertNotEqual(canonical_sha256(before), canonical_sha256(after))
        self.assertEqual(owners["custom"]["revision"], 1)

    def test_legacy_owner_contract_preserves_payload_and_digest(self):
        class FactorizedOwner(nn.Module):
            fsq_levels = (2, 3)
            codebook_sizes = (6,)
            num_embeddings = 6
            embedding_dim = 2
            radix_order = "little"

            def __init__(self):
                super().__init__()
                self.register_buffer("topology", torch.tensor([[0, 1]]))
                self.register_buffer("level_values", torch.tensor([0.0, 1.0]))

            @property
            def weight(self):
                raise AssertionError("legacy factorized owner must not materialize weight")

        bank = EmbeddingBank(2)
        bank.add_owner("factorized", FactorizedOwner(), state={"family": "custom"})
        bank.add_owner("dense", nn.Embedding(3, 2))
        expected = {
            "grammar": "embedding-bank-v1",
            "embedding_dim": 2,
            "logical_to_owner": {"factorized": "factorized", "dense": "dense"},
            "owners": {
                "factorized": {
                    "family": "custom",
                    "implementation": "fsq-factorized-v1",
                    "rows": 6,
                    "embedding_dim": 2,
                    "fsq_levels": [2, 3],
                    "codebook_sizes": [6],
                    "radix_order": "little",
                    "topology": [[0, 1]],
                    "level_values_sha256": canonical_sha256([0.0, 1.0]),
                },
                "dense": {
                    "implementation": "dense-v1",
                    "family": "dense",
                    "rows": 3,
                    "embedding_dim": 2,
                },
            },
        }
        self.assertEqual(bank.contract_state(), expected)
        self.assertEqual(canonical_sha256(bank.contract_state()), canonical_sha256(expected))

    def test_owner_provider_rejects_invalid_contract(self):
        class InvalidOwner(nn.Module):
            def __init__(self, provider):
                super().__init__()
                self.embedding_contract_state = provider

        for provider in (42, lambda: [], lambda: {"invalid": object()}):
            with self.subTest(provider=provider):
                bank = EmbeddingBank()
                bank.add_owner("invalid", InvalidOwner(provider))
                with self.assertRaises(TypeError):
                    bank.contract_state()


if __name__ == "__main__":
    unittest.main()
