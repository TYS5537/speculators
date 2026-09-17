"""Execute the real checkpoint dispatch loop without importing PyTorch."""

# ruff: noqa: PT009 -- Stdlib-only control-flow regression suite.

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/speculators/models/dflash/core.py"


def _load_dispatch(namespace):
    """Keep the production loop intact; substitute only its surrounding tensors."""
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    model = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashDraftModel"
    )
    methods = {
        node.name: node for node in model.body if isinstance(node, ast.FunctionDef)
    }
    loop = next(
        node for node in methods["_backbone_forward"].body if isinstance(node, ast.For)
    )
    wrapper = ast.parse(
        "def dispatch(self, noise_embedding, fc_output, full_attn_mask, "
        "sliding_window_attn_mask, position_ids, position_embeddings, **kwargs):\n"
        "    pass\n"
    ).body[0]
    wrapper.body = [
        loop,
        ast.Return(value=ast.Name(id="noise_embedding", ctx=ast.Load())),
    ]
    default_policy = next(
        node
        for node in methods["__init__"].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "activation_checkpointing"
            for target in node.targets
        )
    )
    initializer = ast.parse("def initialize(self):\n    pass\n").body[0]
    initializer.body = [default_policy]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            initializer,
            methods["set_activation_checkpointing"],
            wrapper,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- Only current-checkout definitions, never external input.
        compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace
    )
    return namespace


class CheckpointDispatchTests(unittest.TestCase):
    def setUp(self):
        self.grad_enabled = True
        self.checkpoint_calls = []

        def checkpoint(module, *args, **kwargs):
            self.checkpoint_calls.append((module, args, dict(kwargs)))
            self.assertIs(kwargs.pop("use_reentrant"), False)
            self.assertIs(kwargs.pop("preserve_rng_state"), True)
            return module(*args, **kwargs)

        self.namespace = _load_dispatch(
            {
                "torch": SimpleNamespace(is_grad_enabled=lambda: self.grad_enabled),
                "checkpoint": checkpoint,
            }
        )
        self.layers = [
            Mock(
                side_effect=lambda target_hidden, hidden_states, **kwargs: (
                    0,
                    hidden_states,
                )
            ),
            Mock(
                side_effect=lambda target_hidden, hidden_states, **kwargs: (
                    1,
                    hidden_states,
                )
            ),
        ]
        self.model = SimpleNamespace(
            layers=self.layers, training=True, sliding_window_indices=[1]
        )
        self.namespace["initialize"](self.model)
        self.inputs = {
            "noise_embedding": object(),
            "fc_output": object(),
            "full_attn_mask": object(),
            "sliding_window_attn_mask": object(),
            "position_ids": object(),
            "position_embeddings": (object(), object()),
            "custom_option": object(),
        }

    def run_dispatch(self):
        return self.namespace["dispatch"](self.model, **self.inputs)

    def test_default_and_setter_touch_runtime_policy_only(self):
        self.assertIs(self.model.activation_checkpointing, False)
        before = dict(vars(self.model))
        for enabled in (True, False):
            self.namespace["set_activation_checkpointing"](self.model, enabled)
            self.assertEqual(
                vars(self.model), {**before, "activation_checkpointing": enabled}
            )

    def test_enabled_uses_each_module_directly_and_preserves_all_kwargs(self):
        self.model.activation_checkpointing = True
        output = self.run_dispatch()
        self.assertEqual(output, (1, (0, self.inputs["noise_embedding"])))
        self.assertEqual(len(self.checkpoint_calls), 2)
        for index, (module, args, kwargs) in enumerate(self.checkpoint_calls):
            self.assertIs(module, self.layers[index])
            self.assertIs(kwargs["use_cache"], False)
            self.assertEqual(len(args), 2)
            self.assertIs(args[0], self.inputs["fc_output"])
            self.assertNotIn("target_hidden", kwargs)
            self.assertNotIn("hidden_states", kwargs)
            self.assertIs(kwargs["position_ids"], self.inputs["position_ids"])
            self.assertIs(
                kwargs["position_embeddings"], self.inputs["position_embeddings"]
            )
            self.assertIs(kwargs["custom_option"], self.inputs["custom_option"])
            expected_mask = "sliding_window_attn_mask" if index else "full_attn_mask"
            self.assertIs(kwargs["attention_mask"], self.inputs[expected_mask])
        self.assertIs(self.checkpoint_calls[0][1][1], self.inputs["noise_embedding"])
        self.assertEqual(
            self.checkpoint_calls[1][1][1],
            (0, self.inputs["noise_embedding"]),
        )

    def test_only_enabled_training_with_grad_uses_checkpoint(self):
        for enabled in (False, True):
            for training in (False, True):
                for grad_enabled in (False, True):
                    with self.subTest(
                        enabled=enabled, training=training, grad_enabled=grad_enabled
                    ):
                        self.checkpoint_calls.clear()
                        self.model.activation_checkpointing = enabled
                        self.model.training = training
                        self.grad_enabled = grad_enabled
                        self.run_dispatch()
                        self.assertEqual(
                            len(self.checkpoint_calls),
                            2 if enabled and training and grad_enabled else 0,
                        )


if __name__ == "__main__":
    unittest.main()
