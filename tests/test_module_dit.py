import unittest
from unittest.mock import patch

import torch

from anytrain.module import DiT, DiTAttentionBackend, DiTConditionState, DiTConditionType


class DiTModuleTest(unittest.TestCase):
    def test_none_condition_rejects_condition_inputs(self):
        model = DiT(input_dim=5, hidden_dim=8, layers=1, heads=2, ffn_ratio=2)
        x_t = torch.randn(2, 3, 5)
        t = torch.tensor([0.2, 0.4])

        output = model(x_t, t)

        self.assertEqual(output.shape, x_t.shape)
        with self.assertRaisesRegex(ValueError, "does not accept"):
            model.prepare_condition(torch.randn(2, 3, 4))

    def test_frame_film_condition_trains_and_masks_output(self):
        model = DiT(
            input_dim=5,
            output_dim=7,
            hidden_dim=12,
            layers=2,
            heads=3,
            ffn_ratio=2,
            condition_dim=4,
            condition_type=DiTConditionType.FRAME_FILM,
            attention_backend=DiTAttentionBackend.EAGER,
        )
        x_t = torch.randn(2, 4, 5)
        t = torch.tensor([0.1, 0.8])
        condition = torch.randn(2, 4, 4)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

        output = model(x_t, t, mask=mask, condition=condition)
        output.square().mean().backward()

        self.assertEqual(output.shape, (2, 4, 7))
        self.assertTrue(torch.equal(output[1, 2:], torch.zeros_like(output[1, 2:])))
        self.assertIsNotNone(model.input.weight.grad)
        self.assertTrue(torch.isfinite(model.input.weight.grad).all())

    def test_vector_film_condition_broadcasts(self):
        model = DiT(
            input_dim=5,
            hidden_dim=8,
            layers=1,
            heads=2,
            ffn_ratio=2,
            condition_dim=4,
            condition_type="film",
        )
        x_t = torch.randn(2, 3, 5)
        t = torch.tensor([0.2, 0.4])
        condition = torch.randn(2, 4)

        state = model.prepare_condition(condition)
        output = model(x_t, t, condition_state=state)

        self.assertIsInstance(state, DiTConditionState)
        self.assertEqual(state.film.shape, (2, 1, 8))
        self.assertEqual(output.shape, x_t.shape)

    def test_cross_attention_condition_state_reuses_cached_kv(self):
        torch.manual_seed(0)
        model = DiT(
            input_dim=5,
            hidden_dim=8,
            layers=2,
            heads=2,
            ffn_ratio=2,
            condition_dim=6,
            condition_type=DiTConditionType.CROSS_ATTN,
            attention_backend=DiTAttentionBackend.SDPA,
        )
        x_t = torch.randn(2, 4, 5)
        t = torch.tensor([0.1, 0.8])
        condition = torch.randn(2, 3, 6)
        condition_mask = torch.tensor([[True, True, False], [True, False, False]])

        state = model.prepare_condition(condition, condition_mask=condition_mask)
        cached = model(x_t, t, condition_state=state)
        raw = model(x_t, t, condition=condition, condition_mask=condition_mask)

        self.assertEqual(len(state.cross_kv), 2)
        self.assertEqual(state.cross_kv[0].key.shape, (2, 2, 3, 4))
        torch.testing.assert_close(cached, raw)

    def test_forward_with_features_uses_selected_layer(self):
        model = DiT(
            input_dim=5,
            hidden_dim=8,
            layers=2,
            heads=2,
            ffn_ratio=2,
            condition_dim=4,
            condition_type=DiTConditionType.FRAME_FILM,
            feature_dim=6,
            feature_layer=1,
        )
        x_t = torch.randn(2, 3, 5)
        t = torch.tensor([0.2, 0.4])
        condition = torch.randn(2, 3, 4)

        output, features = model.forward_with_features(x_t, t, condition=condition)

        self.assertEqual(output.shape, x_t.shape)
        self.assertEqual(features.shape, (2, 3, 6))

    def test_trusted_forward_skips_row_validation(self):
        model = DiT(
            input_dim=5,
            hidden_dim=8,
            layers=1,
            heads=2,
            ffn_ratio=2,
            condition_dim=4,
            condition_type=DiTConditionType.FRAME_FILM,
        )
        x_t = torch.randn(2, 3, 5)
        t = torch.tensor([0.2, 0.4])
        condition = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, True, True], [True, False, False]])

        with patch(
            "anytrain.module.dit._require_valid_row",
            side_effect=AssertionError("trusted path should skip validation"),
        ):
            output = model(x_t, t, condition=condition, mask=mask, validate=False)

        self.assertEqual(output.shape, x_t.shape)

    def test_validates_public_shapes_and_state_contract(self):
        model = DiT(
            input_dim=5,
            hidden_dim=8,
            layers=1,
            heads=2,
            ffn_ratio=2,
            condition_dim=4,
            condition_type=DiTConditionType.FRAME_FILM,
        )
        x_t = torch.randn(2, 3, 5)
        t = torch.tensor([0.2, 0.4])

        with self.assertRaisesRegex(TypeError, "boolean"):
            model(x_t, t, mask=torch.ones(2, 3), condition=torch.randn(2, 3, 4))
        with self.assertRaisesRegex(ValueError, "align"):
            model(x_t, t, condition=torch.randn(2, 4, 4))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            state = model.prepare_condition(torch.randn(2, 3, 4))
            model(x_t, t, condition=torch.randn(2, 3, 4), condition_state=state)

    def test_cross_attention_validates_condition_mask(self):
        model = DiT(
            input_dim=5,
            hidden_dim=8,
            layers=1,
            heads=2,
            ffn_ratio=2,
            condition_dim=6,
            condition_type=DiTConditionType.CROSS_ATTN,
        )

        with self.assertRaisesRegex(ValueError, "valid"):
            model.prepare_condition(
                torch.randn(2, 3, 6),
                condition_mask=torch.tensor([[True, False, False], [False, False, False]]),
            )


if __name__ == "__main__":
    unittest.main()
