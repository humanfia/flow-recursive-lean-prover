"""The stand-in `humanize1` the flow's tests run against, held to the real one it is pinned to.

`test_recursive_lean_prover.py` runs the flow beside a stand-in that declares the interface of
`humanize1:gen-plan` and `humanize1:rlcr`. This holds that stand-in to a checkout of
humanfia/humanize1-flow at the tag the flow is pinned to, put beside the flow as hmz installs
the two, and resolved from inside the flow by the refs it calls them by. The checkout is
named by `HUMANIZE1_CHECKOUT`, which CI fetches before the tests run; without one this is
skipped.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from hmz.runtime.flowing import loading
from hmz.runtime.flowing.engine import load_flow

from tests.test_recursive_lean_prover import FLOW, HUMANIZE1

CHECKOUT = os.environ.get("HUMANIZE1_CHECKOUT", "")

pytestmark = pytest.mark.skipif(
    not CHECKOUT,
    reason="HUMANIZE1_CHECKOUT names no checkout of humanfia/humanize1-flow at its pinned tag",
)

IGNORED = shutil.ignore_patterns("__pycache__")


def interface(flow: Any) -> dict[str, Any]:
    """What a caller of a flow depends on: whether it resumes, what each role must be, and
    each param's type and default."""
    declared = flow.describe()
    return {
        "resumable": declared.resumable,
        "agents": [
            (
                role.name,
                role.auto,
                role.required,
                role.harness,
                sorted(one.__qualname__ for one in role.capabilities),
                role.permission,
            )
            for role in declared.agents
        ],
        "envs": [
            (
                role.name,
                role.auto,
                sorted(one.__qualname__ for one in role.capabilities),
            )
            for role in declared.envs
        ],
        "params": {
            name: (repr(field.annotation), field.get_default(call_default_factory=True))
            for name, field in declared.params.model_fields.items()
        },
    }


def test_the_stand_in_declares_what_the_pinned_humanize1_does(tmp_path: Path) -> None:
    stand_in = tmp_path / "stand-in"
    (stand_in / "humanize1").mkdir(parents=True)
    (stand_in / "humanize1" / "__init__.py").write_text(textwrap.dedent(HUMANIZE1))
    try:
        expected = {
            name: interface(
                load_flow(f"{stand_in / 'humanize1'}:{name}", caller_globals={})
            )
            for name in ("gen-plan", "rlcr")
        }
    finally:
        loading.forget(stand_in)

    flows = tmp_path / "flows"
    shutil.copytree(FLOW, flows / FLOW.name, ignore=IGNORED)
    shutil.copytree(Path(CHECKOUT) / "humanize1", flows / "humanize1", ignore=IGNORED)
    try:
        entry = load_flow(str(flows / FLOW.name), caller_globals={})
        callers = {
            "gen-plan": vars(sys.modules["_recursive_lean.runtime"]),
            "rlcr": vars(sys.modules[entry.fn.__module__]),
        }
        for name, caller in callers.items():
            found = loading.load(f"humanize1:{name}", caller)
            assert (
                Path(found.fn.__code__.co_filename)
                .resolve()
                .is_relative_to((flows / "humanize1").resolve())
            ), f"humanize1:{name} did not resolve to the sibling checkout"
            assert interface(found) == expected[name]
    finally:
        loading.forget(flows)
