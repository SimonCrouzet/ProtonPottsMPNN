"""Import the PH engine module without its heavy third-party stack.

The engine module imports atomworks, biotite and friends at module level, although the
code under test (scorer, objective helpers, block descent) is pure torch. This helper
stubs *only the packages that are not installed*, so the same tests run against the real
packages once the environment exists, and stubs nothing then.

Stubbed CamelCase names are plain classes (so project classes can subclass them); other
names are ``MagicMock`` objects. Nothing the stubs return may be relied on by a test.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ENGINE_MODULE = "mpnn.inference_engines.potts_mpnn_ph"
OPTIONAL_ROOTS = (
    "atomworks",
    "biotite",
    "lightning",
    "hydra",
    "omegaconf",
    "sklearn",
    "flaml",
    "tqdm",
    "propka",
    "rich",
    "beartype",
    "jaxtyping",
    "einops",
    "environs",
    "ipdb",
)
FOUNDRY_SRC = Path(__file__).resolve().parents[4] / "src"


def _stub_class(name: str) -> type:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return MagicMock(name=f"{name}.{attr}")

    return type(name, (), {"__init__": __init__, "__getattr__": __getattr__})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[:1].isupper() and not name.isupper():
            value = _stub_class(name)
        else:
            value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


@contextlib.contextmanager
def engine_module():
    """Yield the imported engine module; undo all stubbing and imports on exit."""
    before = set(sys.modules)
    path_before = list(sys.path)
    missing = [r for r in OPTIONAL_ROOTS if importlib.util.find_spec(r) is None]
    finder = _StubFinder(missing)
    if importlib.util.find_spec("foundry") is None and FOUNDRY_SRC.exists():
        sys.path.append(str(FOUNDRY_SRC))
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module(ENGINE_MODULE)
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = path_before
        # Drop only what this import brought in from the project and the stubs: removing
        # real third-party modules (torch internals) would make a re-import fail.
        owned = {"mpnn", "foundry", *missing}
        for name in set(sys.modules) - before:
            if name.split(".")[0] in owned:
                del sys.modules[name]
