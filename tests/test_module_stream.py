import unittest

import torch
from torch import Tensor, nn

from anytrain.module.stream import (
    Align,
    AlignMerge,
    AlignMethod,
    AlignSpec,
    Codes,
    ControlTokens,
    Features,
    Flatten,
    FrameMajor,
    Materializer,
    Stream,
    StreamMajor,
    Timeline,
    bind,
    bind_targets,
    compile_layout,
)


def _timeline(
    lengths: list[int], size: int | None = None, frame_rate: float | None = None
) -> Timeline:
    return Timeline(
        torch.tensor(lengths), max(lengths, default=0) if size is None else size, frame_rate
    )


class ValueEmbedder(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        if values.ndim == 2:
            return values.to(dtype=torch.float32).unsqueeze(-1)
        return values.to(dtype=torch.float32).sum(dim=-1, keepdim=True)


class CountingEmbedder(ValueEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, values: Tensor) -> Tensor:
        self.calls += 1
        return super().forward(values)


class StreamLayoutTest(unittest.TestCase):
    def test_layout_compiles_before_binding_and_binding_can_be_partial(self):
        expression = Flatten(
            "flat",
            (StreamMajor((Stream("a"), Stream("b"), Stream("a"))),),
        )
        layout = compile_layout(
            expression,
            {"a": _timeline([2]), "b": _timeline([1])},
        )

        partial = bind(
            layout,
            {"a": Codes("a", torch.tensor([[1, 2]]), torch.tensor([2]))},
        )
        self.assertEqual(partial.missing, ("b",))
        complete = partial.bind(
            {
                "b": Features(
                    "b",
                    torch.tensor([[[3.0]]]),
                    torch.tensor([1]),
                )
            }
        )
        self.assertTrue(complete.complete)
        with self.assertRaisesRegex(ValueError, "already bound"):
            partial.bind({"a": Codes("a", torch.tensor([[4, 5]]), torch.tensor([2]))})

    def test_codes_embed_align_then_flatten(self):
        align = Align(
            "bc",
            "b",
            "c",
            AlignSpec(AlignMethod.LINEAR, AlignMerge.ADD),
        )
        expression = Flatten(
            "input",
            (StreamMajor(("a", align)),),
        )
        layout = compile_layout(
            expression,
            {
                "a": _timeline([2], frame_rate=20.0),
                "b": _timeline([4], frame_rate=40.0),
                "c": _timeline([2], frame_rate=20.0),
            },
        )
        bound = bind(
            layout,
            {
                "a": Codes("a", torch.tensor([[1, 2]]), torch.tensor([2]), 20.0),
                "b": Codes("b", torch.tensor([[100, 100, 100, 100]]), torch.tensor([4]), 40.0),
                "c": Codes("c", torch.tensor([[0, 10]]), torch.tensor([2]), 20.0),
            },
        )

        output = Materializer(
            embedders={"a": ValueEmbedder(), "b": ValueEmbedder(), "c": ValueEmbedder()}
        )(bound)

        torch.testing.assert_close(
            output.values,
            torch.tensor([[[1.0], [2.0], [100.0], [102.5], [107.5], [110.0]]]),
        )
        self.assertEqual(output.lengths.tolist(), [6])
        self.assertEqual(layout.owner.tolist(), [[0, 0, 1, 1, 1, 1]])
        self.assertEqual(layout.source_index.tolist(), [[0, 1, 0, 1, 2, 3]])

        targets = bind_targets(
            layout,
            {
                "a": bound.values["a"],
                "b": bound.values["b"],
            },
        )
        self.assertEqual(targets.values.tolist(), [[1, 2, 100, 100, 100, 100]])
        with self.assertRaisesRegex(ValueError, "non-carrier"):
            bind_targets(
                layout, {"a": bound.values["a"], "b": bound.values["b"], "c": bound.values["c"]}
            )

    def test_direct_features_can_use_pool_then_add(self):
        expression = Align(
            "frame_with_fixed",
            "frame",
            "fixed",
            AlignSpec(AlignMethod.BROADCAST, AlignMerge.ADD),
        )
        layout = compile_layout(
            expression,
            {
                "frame": _timeline([3], frame_rate=30.0),
                "fixed": _timeline([2], size=3),
            },
        )
        output = Materializer()(
            bind(
                layout,
                {
                    "frame": Features("frame", torch.zeros(1, 3, 1), torch.tensor([3]), 30.0),
                    "fixed": Features(
                        "fixed",
                        torch.tensor([[[2.0], [4.0], [100.0]]]),
                        torch.tensor([2]),
                    ),
                },
            )
        )
        torch.testing.assert_close(output.values, torch.full((1, 3, 1), 3.0))

    def test_nested_flatten_preserves_leaf_lineage(self):
        inner = Flatten("bc", (StreamMajor(("b", "c")),))
        expression = Flatten("all", (StreamMajor(("a", inner)),))
        layout = compile_layout(
            expression,
            {"a": _timeline([1]), "b": _timeline([2]), "c": _timeline([1])},
        )
        bound = bind(
            layout,
            {
                "a": Codes("a", torch.tensor([[1]]), torch.tensor([1])),
                "b": Codes("b", torch.tensor([[2, 3]]), torch.tensor([2])),
                "c": Codes("c", torch.tensor([[4]]), torch.tensor([1])),
            },
        )
        output = Materializer(embedders={name: ValueEmbedder() for name in ("a", "b", "c")})(bound)
        torch.testing.assert_close(
            output.values,
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        )
        self.assertEqual(output.layout.owner.tolist(), [[0, 1, 1, 2]])
        self.assertEqual(output.layout.source_index.tolist(), [[0, 0, 1, 0]])

    def test_frame_major_interleaves_rvq_stages(self):
        expression = Flatten("rvq", (FrameMajor(("q0", "q1")),))
        layout = compile_layout(
            expression,
            {
                "q0": _timeline([2], frame_rate=50.0),
                "q1": _timeline([2], frame_rate=50.0),
            },
        )
        q0 = Codes("q0", torch.tensor([[10, 11]]), torch.tensor([2]), 50.0)
        q1 = Codes("q1", torch.tensor([[20, 21]]), torch.tensor([2]), 50.0)
        output = Materializer(embedders={"q0": ValueEmbedder(), "q1": ValueEmbedder()})(
            bind(layout, {"q0": q0, "q1": q1})
        )
        torch.testing.assert_close(
            output.values,
            torch.tensor([[[10.0], [20.0], [11.0], [21.0]]]),
        )
        self.assertEqual(output.layout.owner.tolist(), [[0, 1, 0, 1]])
        self.assertEqual(output.layout.source_index.tolist(), [[0, 0, 1, 1]])

    def test_frame_major_requires_equal_lengths_and_frame_rates(self):
        expression = Flatten("rvq", (FrameMajor(("q0", "q1")),))
        with self.assertRaisesRegex(ValueError, "equal per-sample lengths"):
            compile_layout(expression, {"q0": _timeline([2]), "q1": _timeline([1])})
        with self.assertRaisesRegex(ValueError, "equal frame rates"):
            compile_layout(
                expression,
                {"q0": _timeline([2], frame_rate=50.0), "q1": _timeline([2], frame_rate=25.0)},
            )

    def test_frame_major_segment_and_frame_controls_are_materialized_by_name(self):
        controls = ControlTokens(
            segment_start="segment:start",
            segment_end="segment:end",
            frame_start="frame:start",
            frame_end="frame:end",
        )
        expression = Flatten("flat", (FrameMajor(("a",), controls),))
        layout = compile_layout(expression, {"a": _timeline([2])})
        self.assertEqual(
            layout.control_names,
            ("segment:start", "frame:start", "frame:end", "segment:end"),
        )
        codes = Codes("a", torch.tensor([[7, 8]]), torch.tensor([2]))
        embedding = nn.Embedding.from_pretrained(torch.arange(10, dtype=torch.float32).view(10, 1))
        output = Materializer(
            embedders={"a": ValueEmbedder()},
            control_embedding=embedding,
            control_ids={
                "segment:start": 9,
                "segment:end": 6,
                "frame:start": 5,
                "frame:end": 4,
            },
        )(bind(layout, {"a": codes}))
        torch.testing.assert_close(
            output.values,
            torch.tensor([[[9.0], [5.0], [7.0], [4.0], [5.0], [8.0], [4.0], [6.0]]]),
        )
        targets = bind_targets(layout, {"a": codes}, predicted_controls=("frame:start",))
        self.assertEqual(targets.owner.tolist(), [[-1, -1, 0, -1, -1, 0, -1, -1]])
        self.assertEqual(targets.values.tolist(), [[-100, 1, 7, -100, 1, 8, -100, -100]])
        self.assertEqual(
            targets.predict_mask.tolist(), [[False, True, True, False, True, True, False, False]]
        )

    def test_control_namespace_does_not_depend_on_active_frames(self):
        controls = ControlTokens(frame_start="frame:start", frame_end="frame:end")
        expression = Flatten("flat", (FrameMajor(("a",), controls),))
        layout = compile_layout(
            expression,
            {"a": Timeline(torch.empty(0, dtype=torch.long), 0)},
        )
        self.assertEqual(layout.control_names, ("frame:start", "frame:end"))
        output = Materializer()(
            bind(
                layout,
                {"a": Features("a", torch.empty(0, 0, 1), torch.empty(0, dtype=torch.long))},
            )
        )
        self.assertEqual(tuple(output.values.shape), (0, 0, 1))

    def test_structured_codes_can_be_embedded_without_layout_knowledge(self):
        expression = Stream("rvq")
        layout = compile_layout(expression, {"rvq": _timeline([2])})
        codes = Codes("rvq", torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([2]))
        output = Materializer(embedders={"rvq": ValueEmbedder()})(bind(layout, {"rvq": codes}))
        torch.testing.assert_close(output.values, torch.tensor([[[3.0], [7.0]]]))

    def test_shared_stream_is_embedded_once_in_the_dag(self):
        c = Stream("c")
        aligned = Align("bc", "b", c, AlignSpec(AlignMethod.IDENTITY, AlignMerge.SOURCE))
        expression = Flatten("flat", (StreamMajor((aligned, c, c)),))
        layout = compile_layout(expression, {"b": _timeline([2]), "c": _timeline([2])})
        c_embedder = CountingEmbedder()
        output = Materializer(embedders={"b": ValueEmbedder(), "c": c_embedder})(
            bind(
                layout,
                {
                    "b": Codes("b", torch.tensor([[1, 2]]), torch.tensor([2])),
                    "c": Codes("c", torch.tensor([[3, 4]]), torch.tensor([2])),
                },
            )
        )
        self.assertEqual(c_embedder.calls, 1)
        torch.testing.assert_close(
            output.values,
            torch.tensor([[[3.0], [4.0], [3.0], [4.0], [3.0], [4.0]]]),
        )

    def test_identity_alignment_requires_equal_lengths(self):
        expression = Align("same", "target", "source", AlignSpec(AlignMethod.IDENTITY))
        with self.assertRaisesRegex(ValueError, "equal sample lengths"):
            compile_layout(expression, {"source": _timeline([3]), "target": _timeline([2])})

    def test_empty_source_is_allowed_when_every_target_frame_is_padding(self):
        expression = Align("same", "target", "source", AlignSpec(AlignMethod.IDENTITY))
        layout = compile_layout(
            expression,
            {
                "source": Timeline(torch.tensor([0]), 0),
                "target": Timeline(torch.tensor([0]), 3),
            },
        )
        output = Materializer()(
            bind(
                layout,
                {
                    "source": Features("source", torch.empty(1, 0, 1), torch.tensor([0])),
                    "target": Features("target", torch.ones(1, 3, 1), torch.tensor([0])),
                },
            )
        )
        torch.testing.assert_close(output.values, torch.zeros(1, 3, 1))


if __name__ == "__main__":
    unittest.main()
