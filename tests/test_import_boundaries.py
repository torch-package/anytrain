"""Core and domain contracts must not import the Lightning integration layer."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "anytrain.module.embedding",
        "anytrain.module.dirichlet_tempering",
        "anytrain.module.quantization.embedding",
        "anytrain.evaluator.text",
        "anytrain.evaluator.translation.comet",
        "anytrain.s2s",
    ],
)
def test_contract_imports_without_lightning(module: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys

class NoLightning(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'lightning' or fullname.startswith(('lightning.', 'anytrain.lightning')):
            raise AssertionError(f'unexpected integration import: {fullname}')
        return None

sys.meta_path.insert(0, NoLightning())
importlib.import_module(sys.argv[1])
""",
            module,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
