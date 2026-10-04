"""recursive_lean_prover, run through the engine on fakes over real git repositories.

The flow is copied into a flowverse of its own beside a stand-in `humanize1` that declares the
pinned `gen-plan` and `rlcr` interface, so that what it calls is resolved by the real loader
and recorded, without depending on the real `humanize1`. Its workspace is a directory on this
machine, served by `DirEnv` -- a small environment driver over asyncio subprocesses -- so git
worktrees, cherry-picks and the comparator are the real thing; the agents are scripted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pydantic
import pytest
from hmz.flows import (
    EnvBackendKind,
    EnvCommandTimeout,
    EnvError,
    FilesEnvMixin,
    GitWorktreeEnvMixin,
    HarnessKind,
    ParamsError,
    Permission,
    PermissionRequestHookAgentMixin,
    RequirementError,
    ScratchDirEnvMixin,
    WorktreeError,
)
from hmz.runtime.flowing import loading
from hmz.runtime.flowing.engine import load_flow
from hmz.runtime.flowing.fakes import FakeAgentDriver, run_fake
from hmz.runtime.flowing.skills import brought
from hmz.runtime.flowing.spi import ENV_CAPABILITIES, Placement

FLOW = Path(__file__).parents[1] / "recursive_lean_prover"

#: A stand-in for `humanize1`, declaring exactly the interface this flow is pinned to, and
#: recording every call it answers.
HUMANIZE1 = """
from typing import Literal

from hmz.flows import (
    Agent, AgentCollection, EnvCollection, FilesEnvMixin, FlowParams, LocalEnv, Outworlder,
    PermissionRequestHookAgentMixin, ShellEnvMixin, flow,
)

CALLS = []


class Here(LocalEnv, ShellEnvMixin, FilesEnvMixin): ...


class Builder(Agent, PermissionRequestHookAgentMixin): ...


class Planning(AgentCollection):
    planner: Agent
    analyst: Agent


class Building(AgentCollection):
    builder: Builder
    reviewer: Agent
    human: Outworlder


class Envs(EnvCollection):
    workspace: Here


class Plan(FlowParams):
    input: str = ""
    output: str = ""
    mode: Literal["discussion", "direct"] = "discussion"
    auto_start_rlcr_if_converged: bool = False
    alternative_plan_language: str = ""
    turn_timeout: float = 3600
    total_timeout: float = 14400
    turn_retries: int = 1


class Rlcr(FlowParams):
    plan_file: str = ""
    max: int = 42
    codex_timeout: int = 5400
    full_review_round: int = 5
    base_branch: str = ""
    skip_code_review: bool = False
    track_plan_file: bool = False
    push_every_round: bool = False
    skip_impl: bool = False
    claude_answer_codex: bool = False
    agent_teams: bool = False
    skip_quiz: bool = False
    yolo: bool = False
    privacy: bool = False
    require_bitlesson_entry_for_none: bool = False


def record(name, task, agents, envs, params):
    CALLS.append(
        {
            "flow": name,
            "task": task,
            "agents": {
                role: agents[role].model for role in agents if role != "human"
            },
            "human": "human" in agents,
            "workspace": str(envs["workspace"].workdir),
            "params": params.model_dump(),
        }
    )


@flow(agents=Planning, envs=Envs, params=Plan, name="gen-plan")
async def gen_plan(task, *, agents, envs, params, ctx):
    record("gen-plan", task, agents, envs, params)
    planner, workspace = agents["planner"], envs["workspace"]
    plan = await planner.run(task, session=await planner.spawn(env=workspace))
    await workspace.write(params.output, plan.encode())
    return str(workspace.workdir / params.output)


@flow(agents=Building, envs=Envs, params=Rlcr, name="rlcr", resumable=True)
async def rlcr(task, *, agents, envs, params, ctx):
    record("rlcr", task, agents, envs, params)
    builder = agents["builder"]
    await builder.run(task, session=await builder.spawn(env=envs["workspace"]))
    return "complete"
"""

COMPARATOR = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$HUMANIZE_NODE_ID" >> "$HUMANIZE_RUN_DIR/compared.txt"
printf '%s\\n' 'Your solution is okay!'
"""


# ------------------------------------------------------------------------ the environment


class DirEnv:
    """A directory on this machine, as an environment driver serving every capability.

    What fills the flow's `workspace`: commands run in the directory through asyncio
    subprocesses, worktrees are `git worktree add --detach`, and scratch directories are kept
    under `scratch`. Everything it derives shares its command log.
    """

    backend = EnvBackendKind.LOCAL
    provider = ""
    capabilities = ENV_CAPABILITIES
    cpu_count = 8
    memory = 64 << 30
    gpu_count = 0
    gpu_memory = 0
    available = True

    def __init__(
        self, workdir: Path, scratch: Path, log: list[tuple[Path, Any]] | None = None
    ) -> None:
        self.path = Path(workdir)
        self.scratch = Path(scratch)
        self.log = [] if log is None else log

    def __repr__(self) -> str:
        return f"<dir env {self.path}>"

    @property
    def workdir(self) -> PurePosixPath:
        return PurePosixPath(self.path)

    def placement(self) -> Placement:
        return Placement(self.backend, self.provider, self.workdir)

    def _there(self, path: Path) -> DirEnv:
        return DirEnv(path, self.scratch, self.log)

    async def derive_subdir(self, subdir: PurePosixPath | str) -> DirEnv:
        under = PurePosixPath(subdir)
        if under.is_absolute() or ".." in under.parts:
            raise ValueError(f"{subdir} is not under {self.path}")
        (self.path / under).mkdir(parents=True, exist_ok=True)
        return self._there(self.path / under)

    async def exec(
        self,
        argv: Any,
        *,
        timeout: float = 0,
    ) -> tuple[int, str, str]:
        command = ["bash", "-c", argv] if isinstance(argv, str) else list(argv)
        self.log.append((self.path, tuple(command)))
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.path,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            raise EnvError(f"{command[0]}: {error}") from None
        try:
            async with asyncio.timeout(timeout or None):
                out, err = await process.communicate()
        except BaseException as error:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            if isinstance(error, TimeoutError):
                raise EnvCommandTimeout(f"{command!r} ran past {timeout}s") from None
            raise
        return (
            process.returncode or 0,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def read(self, path: str) -> bytes:
        return (self.path / path).read_bytes()

    async def write(self, path: str, data: bytes) -> None:
        target = self.path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def derive_worktree(
        self,
        *,
        ref: str | None = None,
        dir: PurePosixPath | str | None = None,
    ) -> DirEnv:
        target = (
            self.path / dir
            if dir is not None
            else self.scratch / "worktrees" / uuid.uuid4().hex[:8]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        code, out, err = await self.exec(
            ["git", "worktree", "add", "--quiet", "--detach", str(target)]
            + ([ref] if ref else [])
        )
        if code:
            raise WorktreeError(f"git worktree add {target}: {(err or out).strip()}")
        return self._there(target)

    async def derive_temp_clone(self, id: str, *, holder: object) -> DirEnv:
        raise EnvError("DirEnv makes no temporary copies")

    async def destroy_temp_clone(self, id: str) -> None:
        return

    async def derive_scratch(self, id: str) -> DirEnv:
        path = self.scratch / id
        path.mkdir(parents=True, exist_ok=True)
        return self._there(path)

    async def destroy_scratch(self, id: str) -> None:
        shutil.rmtree(self.scratch / id, ignore_errors=True)

    async def close(self) -> None:
        return


# ------------------------------------------------------------------------------ fixtures


def git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def repository(
    project: Path,
    *,
    ignore: str = ".humanize/\n.lake/\n",
    submission: str = "namespace Submission\nend Submission\n",
    author: str = "Flow Test",
) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    git(project, "init", "-b", "main")
    git(project, "config", "user.name", author)
    git(project, "config", "user.email", "flow-test@example.invalid")
    (project / ".gitignore").write_text(ignore)
    (project / "Submission.lean").write_text(submission)
    git(project, "add", ".gitignore", "Submission.lean")
    git(project, "commit", "-m", "test: initialize fixture")
    return project


@contextlib.contextmanager
def flowverse(root: Path) -> Iterator[SimpleNamespace]:
    """The flow and its stand-in `humanize1`, loaded from a flowverse of their own."""
    flows = root / "flows"
    shutil.copytree(
        FLOW,
        flows / "recursive_lean_prover",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (flows / "humanize1").mkdir()
    (flows / "humanize1" / "__init__.py").write_text(textwrap.dedent(HUMANIZE1))
    entry = load_flow(str(flows / "recursive_lean_prover"), caller_globals={})
    rlcr = load_flow(f"{flows / 'humanize1'}:rlcr", caller_globals={})
    try:
        yield SimpleNamespace(
            flows=flows,
            entry=entry,
            nested=load_flow(
                f"{flows / 'recursive_lean_prover'}:worktree-rlcr", caller_globals={}
            ),
            turn=load_flow(
                f"{flows / 'recursive_lean_prover'}:turn", caller_globals={}
            ),
            flow=sys.modules[entry.fn.__module__],
            runtime=sys.modules["_recursive_lean.runtime"],
            models=sys.modules["_recursive_lean.models"],
            store=sys.modules["_recursive_lean.store"],
            prompts=sys.modules["_recursive_lean.prompts"],
            preflight=sys.modules["_recursive_lean.preflight"],
            calls=rlcr.globals["CALLS"],
        )
    finally:
        loading.forget(flows)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    with flowverse(tmp_path_factory.mktemp("verse")) as made:
        yield made


#: A complete reference ledger, citing each snapshot's README by a path relative to it.
REFERENCE_USE = [
    {
        "source": source,
        "queries": ["fixture query"],
        "files": ["README.md"],
        "conclusion": "fixture source was consulted",
    }
    for source in ("TauCeti", "lean-pool", "mathlib-internal")
]


def problem_site_data(problem_id: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "generated_at": "2026-09-08T10:02:54Z",
        "problem": {
            "id": problem_id,
            "title": "Mihăilescu's theorem",
            "statement_revision": 1,
            "module": "LeanEval.NumberTheory.Mihailescu",
            "stable_url": f"problems/{problem_id}/",
        },
    }


def problem_markdown(problem_id: str) -> str:
    return f"""# Mihăilescu's theorem

> Source: [Lean AI formalization leaderboard](https://lean-lang.org/eval/problems/{problem_id}/)
> Crawled: 2026-09-08
> Leaderboard data generated: 2026-09-08T10:02:54Z

## Leaderboard entry

| Field | Value |
| --- | --- |
| Problem id | `{problem_id}` |
| Group | `formalization-evaluation` |
| Statement revision | `1` |
| Module | `LeanEval.NumberTheory.Mihailescu` |

## Problem

The one selected mathematical problem.

## Data limitations

- None relevant to this fixture.
"""


def fetched_problem(loaded: Any, problem_id: str) -> Any:
    return loaded.models.FetchedProblem(
        problem_id=problem_id,
        title="Mihăilescu's theorem",
        source_url=f"https://lean-lang.org/eval/problems/{problem_id}/",
        data_url=f"https://lean-lang.org/eval/site-data/v2/problems/{problem_id}.json",
        generated_at="2026-09-08T10:02:54Z",
        statement_revision=1,
        module="LeanEval.NumberTheory.Mihailescu",
        markdown=problem_markdown(problem_id),
    )


def reference_bundle(loaded: Any, root: Path) -> Any:
    """Three reference snapshots, each holding the README `REFERENCE_USE` cites."""
    paths: dict[str, Path] = {}
    for source in loaded.preflight.REFERENCE_SOURCES:
        path = root / source.directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.md").write_text("fixture\n")
        paths[source.name] = path.resolve()
    return loaded.preflight.ReferenceBundle(
        root=root.resolve(),
        manifest=root.resolve() / "manifest.json",
        paths=paths,
        commits={source: "a" * 40 for source in paths},
    )


@pytest.fixture
def offline(loaded: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Preflight with what it would download already here: the snapshots, and the JSON.

    Every reference library prepared is made where it was asked for, and listed.
    """
    prepared: list[Any] = []

    def prepare(library: Any) -> Any:
        prepared.append(reference_bundle(loaded, library.root))
        return prepared[-1]

    async def site_data(runtime: Any) -> dict[str, Any]:
        return problem_site_data(runtime.problem_id)

    monkeypatch.setattr(loaded.preflight.ReferenceLibrary, "prepare", prepare)
    monkeypatch.setattr(loaded.runtime.Runtime, "_problem_site_data", site_data)
    return prepared


def runtime_for(
    loaded: Any,
    project: Path,
    tmp_path: Path,
    task: str = "fixture theorem",
    **params: Any,
) -> Any:
    """The flow's runtime over a workspace, outside a run, to exercise one piece of it."""
    workspace = DirEnv(project, tmp_path / "scratch")
    return loaded.runtime.Runtime(
        {}, {"workspace": workspace}, task, loaded.flow.Config(**params), {}
    )


# --------------------------------------------------------------------------- the contract


def test_nested_rlcr_does_not_enable_generic_code_review(loaded: Any) -> None:
    config = loaded.flow.WorktreeRlcrConfig(
        plan_file="/tmp/immutable-plan.md",
        base_branch="frozen-post-overlay-base",
    )

    forwarded = loaded.flow._nested_rlcr_config(config)

    assert config.base_branch == "frozen-post-overlay-base"
    assert forwarded["base_branch"] == ""
    assert forwarded["skip_code_review"] is True
    assert forwarded["skip_impl"] is False


def test_nested_rlcr_refuses_an_rlcr_that_cannot_skip_review(
    loaded: Any, tmp_path: Path
) -> None:
    class Older(pydantic.BaseModel):
        plan_file: str = ""
        max: int = 42
        base_branch: str = ""

    class OlderRlcr:
        expected_params = Older

        def __init__(self) -> None:
            self.called = False

        async def __call__(self, *_: Any, **__: Any) -> str:
            self.called = True
            return "complete"

    with pytest.raises(RequirementError, match="too old"):
        loaded.flow._require_explicit_rlcr_review_skip(OlderRlcr())

    # Inside the bridge, before RLCR is handed anything: the run stops, failing closed.
    worktree = repository(tmp_path / "node")
    older = OlderRlcr()
    with (
        patch.object(loaded.flow, "load", return_value=older),
        pytest.raises(RequirementError, match="too old"),
    ):
        asyncio.run(
            run_fake(
                loaded.nested,
                "prove the node",
                agents={
                    "worker": FakeAgentDriver(HarnessKind.CODEX),
                    "reviewer": FakeAgentDriver(HarnessKind.CODEX),
                },
                params={"plan_file": "/tmp/immutable-plan.md"},
                local=DirEnv(worktree, tmp_path / "scratch"),
            )
        )
    assert not older.called


def test_public_recursive_lean_flow_contract(loaded: Any) -> None:
    declared = loaded.entry.describe()
    assert declared.name == "recursive_lean_prover"
    assert declared.resumable
    assert not declared.hidden
    assert declared.params is loaded.flow.Config
    assert [role.name for role in declared.agents] == ["worker", "reviewer"]
    worker, reviewer = declared.agents
    assert PermissionRequestHookAgentMixin in worker.capabilities
    assert PermissionRequestHookAgentMixin not in reviewer.capabilities
    for role in declared.agents:
        assert role.required
        assert role.skills == ("recursive-lean-proof",)
        assert role.permission.covers(Permission())
    [workspace] = declared.envs
    assert workspace.name == "workspace"
    assert workspace.auto
    assert {GitWorktreeEnvMixin, ScratchDirEnvMixin, FilesEnvMixin} <= set(
        workspace.capabilities
    )

    nested = loaded.nested.describe()
    assert nested.name == "worktree-rlcr"
    assert nested.hidden
    assert nested.resumable
    assert nested.params is loaded.flow.WorktreeRlcrConfig
    assert [role.name for role in nested.agents] == ["worker", "reviewer"]
    turn = loaded.turn.describe()
    assert turn.name == "turn"
    assert turn.hidden
    assert [(role.name, role.required) for role in turn.agents] == [
        ("worker", False),
        ("reviewer", False),
    ]
    assert [skill.name for skill in brought(FLOW)] == ["recursive-lean-proof"]


def test_params_are_read_as_command_line_values(loaded: Any) -> None:
    config = loaded.entry.params_of(
        {
            "max_parallel_children": "3",
            "stop_on_child_failure": "false",
            "comparator_command": "bash tools/check.sh {node_id}",
        }
    )
    assert config.max_parallel_children == 3
    assert config.stop_on_child_failure is False
    with pytest.raises(ParamsError, match="max_nodes >= 3"):
        loaded.entry.params_of({"max_depth": "1", "max_nodes": "2"})
    with pytest.raises(ParamsError, match=r"below \.humanize/"):
        loaded.entry.params_of({"artifact_dir": "/tmp/elsewhere"})
    with pytest.raises(ParamsError, match=r"must name a \.lean file"):
        loaded.entry.params_of({"lean_target": "Submission.txt"})
    with pytest.raises(ParamsError, match="one Lean-Eval problem id"):
        loaded.entry.params_of({"problem_id": "../mihailescu"})
    with pytest.raises(ParamsError, match="common system environment variable"):
        loaded.entry.params_of({"huggingface_token_env": "PATH"})
    with pytest.raises(ParamsError, match=r"below \.humanize/"):
        loaded.entry.params_of({"reference_dir": "references"})
    config = loaded.entry.params_of(
        {"problem_id": "mihailescu", "agent_hidden_files": '["Solution.lean"]'}
    )
    assert config.problem_id == "mihailescu"
    assert config.agent_hidden_files == ("Solution.lean",)


def test_agent_hidden_files_require_safe_unique_relative_paths(loaded: Any) -> None:
    config = loaded.flow.Config
    assert config(agent_hidden_files=["Solution.lean"]).agent_hidden_files == (
        "Solution.lean",
    )
    for invalid in (["../Solution.lean"], ["/tmp/Solution.lean"], [".git/config"]):
        with pytest.raises(pydantic.ValidationError):
            config(agent_hidden_files=invalid)
    with pytest.raises(pydantic.ValidationError, match="unique"):
        config(agent_hidden_files=["Solution.lean", "Solution.lean"])


# ------------------------------------------------------------------------------ the store


def test_proved_and_accepted_nodes_reject_regressive_transitions(
    loaded: Any, tmp_path: Path
) -> None:
    store = loaded.store.Store(tmp_path / "run", tmp_path / "wiki", "monotone fixture")
    proved = store.ensure(
        "root.proved-a1",
        parent=None,
        depth=0,
        title="Proved",
        statement="A proved theorem",
    )
    store.update(
        proved.id,
        "proved",
        "accepted",
        candidate_commit="proved-candidate",
        theorems=["Submission.proved"],
    )
    store.update(proved.id, "natural-proof", "must be ignored")
    assert proved.status == "proved"
    assert proved.message == "accepted"

    integrating = store.ensure(
        "root.integrating-a1",
        parent=None,
        depth=0,
        title="Integrating",
        statement="An accepted theorem",
    )
    store.update(
        integrating.id,
        "integrating",
        "accepted candidate",
        candidate_commit="candidate",
    )
    store.update(integrating.id, "planning", "must be ignored")
    assert integrating.status == "integrating"
    assert integrating.message == "accepted candidate"


def test_mermaid_arrows_point_from_dependent_to_dependency(
    loaded: Any, tmp_path: Path
) -> None:
    store = loaded.store.Store(tmp_path / "run", tmp_path / "wiki", "diagram fixture")
    store.ensure("root", parent=None, depth=0, title="Root", statement="Root theorem")
    store.ensure(
        "root.base-a1",
        parent="root",
        depth=1,
        title="Base",
        statement="Base theorem",
    )
    store.ensure(
        "root.after-a1",
        parent="root",
        depth=1,
        title="After",
        statement="Dependent theorem",
        depends_on=["root.base-a1"],
    )

    diagram = (tmp_path / "run" / "dag.mmd").read_text()
    assert "Every solid arrow A --&gt; B means A depends on B" in diagram
    assert "n_root --> n_root_base_a1" in diagram
    assert "n_root_after_a1 --> n_root_base_a1" in diagram
    assert "n_root_base_a1 --> n_root_after_a1" not in diagram
    assert "-.->" not in diagram


def test_nested_plan_cannot_reopen_accepted_decomposition(
    loaded: Any, tmp_path: Path
) -> None:
    project = tmp_path / "selected_node_problem"
    project.mkdir()
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "selected node fixture",
        lean_target="Submission.lean",
    )
    node = loaded.models.NodeRecord(
        id="root.selected-a1",
        title="Selected theorem",
        statement="True",
        lean_name="selected",
        lean_statement="True",
        attempts=1,
    )
    scaffold = project / "one-time-plan.md"
    scaffold.write_text("# Historical speculative decomposition\n")
    natural = project / "natural-proof.md"
    natural.write_text("# Accepted mathematical proof\n")
    with patch.object(
        runtime, "_review_command", return_value="bash exact-comparator.sh"
    ):
        implementation = runtime._implementation_plan(
            node,
            accepted_plan=scaffold,
            natural_path=natural,
            children="- `Submission.child`: True",
        ).read_text()

    task = loaded.prompts.RLCR_LEAN_TASK
    assert "Authoritative selected-node contract" in implementation
    assert "override the current DAG" in implementation
    assert "DAG shape, extra certification interface" in implementation
    assert "authoritative implementation boundary" in task
    assert "do not reopen planning or decomposition" in task
    assert "Frozen proof-base commit" in task
    assert "empty list does not ban proof-base helpers" in task


# ------------------------------------------------------------------ worktrees and the lake


def test_node_commit_is_isolated_then_integrated(loaded: Any, tmp_path: Path) -> None:
    project = repository(
        tmp_path / "example_problem", ignore=".humanize/\n.lake/\n/lake-manifest.json\n"
    )
    (project / ".lake" / "packages").mkdir(parents=True)
    (project / "lake-manifest.json").write_text(
        '{"version": "1.1.0", "packages": []}\n'
    )
    runtime = runtime_for(loaded, project, tmp_path)
    node = loaded.models.NodeRecord(
        id="root.leaf-a1", title="Leaf", statement="True", attempts=1
    )

    async def scenario() -> None:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        before = await runtime._git_head(worktree)
        assert path.name == project.name
        assert path.is_relative_to(tmp_path / "scratch" / "recursive-lean-worktrees")
        assert await runtime._git_toplevel(path) == path.resolve()
        assert git(path, "branch", "--show-current") == runtime._node_branch(node)
        original_branch = node.proof_branch
        original_base = node.proof_base_commit
        node.attempts += 1
        again = await runtime._node_worktree(node)
        assert Path(str(again.workdir)) == path
        assert node.proof_branch == original_branch
        assert node.proof_base_commit == original_base
        assert await runtime._git_clean(worktree)
        assert (path / ".lake" / "packages").is_symlink()
        assert (path / "lake-manifest.json").read_text() == (
            project / "lake-manifest.json"
        ).read_text()

        (path / "lake-manifest.json").unlink()
        await runtime._node_worktree(node)
        assert (path / "lake-manifest.json").is_file()

        (path / "Leaf.lean").write_text("theorem leaf : True := by trivial\n")
        git(path, "add", "Leaf.lean")
        git(path, "commit", "-m", "feat: prove leaf")
        after = await runtime._git_head(worktree)

        assert before != after
        assert not (project / "Leaf.lean").exists()
        integrated, feedback = await runtime._integrate_candidate(before, after)
        assert integrated, feedback
        assert (project / "Leaf.lean").is_file()
        assert await runtime._git_clean(runtime.workspace)

    asyncio.run(scenario())


def test_node_worktree_finds_manifest_in_primary_git_worktree(
    loaded: Any, tmp_path: Path
) -> None:
    primary = repository(
        tmp_path / "primary",
        ignore=".humanize/\n.lake/\n/lake-manifest.json\n",
        submission="theorem seed : True := by trivial\n",
    )
    expected = '{"version": "1.1.0", "packages": []}\n'
    (primary / "lake-manifest.json").write_text(expected)
    supervisor = tmp_path / "supervisor"
    git(primary, "worktree", "add", "-b", "supervisor", str(supervisor), "main")
    runtime = runtime_for(loaded, supervisor, tmp_path, "linked manifest fixture")
    node = loaded.models.NodeRecord(
        id="root.linked_manifest-a1",
        title="Linked manifest",
        statement="True",
        attempts=1,
    )

    async def scenario() -> None:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        assert not (supervisor / "lake-manifest.json").exists()
        assert (path / "lake-manifest.json").read_text() == expected
        assert await runtime._git_clean(worktree)

    asyncio.run(scenario())


def test_deep_scratch_paths_fall_back_to_a_hashed_checkout(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(tmp_path / "short_problem")
    deep = tmp_path
    for number in range(3):
        deep /= f"long-experiment-component-{number}-" + "x" * 36
    runtime = loaded.runtime.Runtime(
        {},
        {"workspace": DirEnv(project, deep)},
        "long-path fixture",
        loaded.flow.Config(),
        {},
    )
    node = loaded.models.NodeRecord(
        id="root.long_path_leaf-a1",
        title="Long path leaf",
        statement="True",
        attempts=1,
    )

    async def scenario() -> None:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        root = deep / "recursive-lean-worktrees"
        assert len(str(root / runtime.run_root.name / "long-path-leaf")) > 180
        assert path.parent.parent == root
        assert re.fullmatch(r"[0-9a-f]{12}", path.parent.name)
        assert path.name == project.name
        assert await runtime._git_toplevel(path) == path.resolve()
        assert node.worktree == str(path)

    asyncio.run(scenario())


def test_worktree_rlcr_copies_the_manifest_and_calls_rlcr_in_process(
    loaded: Any, tmp_path: Path
) -> None:
    primary = repository(
        tmp_path / "primary", ignore=".humanize/\n.lake/\n/lake-manifest.json\n"
    )
    expected = '{"version": "1.1.0", "packages": []}\n'
    (primary / "lake-manifest.json").write_text(expected)
    worktree = tmp_path / "node" / "primary"
    git(primary, "worktree", "add", "--detach", str(worktree), "HEAD")
    loaded.calls.clear()
    worker = FakeAgentDriver(HarnessKind.CODEX, model="worker-model", reply="built")
    reviewer = FakeAgentDriver(HarnessKind.CODEX, model="reviewer-model")

    asyncio.run(
        run_fake(
            loaded.nested,
            "prove the node",
            agents={"worker": worker, "reviewer": reviewer},
            params={
                "plan_file": "/tmp/immutable-plan.md",
                "max": 7,
                "base_branch": "frozen-post-overlay-base",
            },
            local=DirEnv(worktree, tmp_path / "scratch"),
        )
    )

    assert (worktree / "lake-manifest.json").read_text() == expected
    [call] = loaded.calls
    assert call["flow"] == "rlcr"
    assert call["task"] == "prove the node"
    assert call["agents"] == {"builder": "worker-model", "reviewer": "reviewer-model"}
    assert call["human"]
    assert call["workspace"] == str(worktree)
    assert call["params"]["plan_file"] == "/tmp/immutable-plan.md"
    assert call["params"]["max"] == 7
    assert call["params"]["base_branch"] == ""
    assert call["params"]["skip_code_review"] is True
    assert call["params"]["skip_quiz"] is True
    assert call["params"]["privacy"] is True
    assert call["params"]["claude_answer_codex"] is True
    [session] = worker.sessions
    assert Path(str(session.placement.workdir)) == worktree


def test_comparator_runs_in_its_env_with_node_variables_and_a_timeout(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(tmp_path / "comparator_problem")
    tools = project / "tools"
    tools.mkdir()
    (tools / "check.sh").write_text(
        'printf "%s|%s\\n" "$HUMANIZE_NODE_ID" "$HUMANIZE_LEAN_FILES"\n'
        'printf "%s\\n" "Your solution is okay!"\n'
    )
    (tools / "slow.sh").write_text("sleep 30\n")
    node = loaded.models.NodeRecord(
        id="root.compared-a1", title="Compared", statement="True", attempts=1
    )

    async def scenario() -> None:
        runtime = runtime_for(
            loaded, project, tmp_path, comparator_command="bash tools/check.sh"
        )
        passed, path, log = await runtime._compare(node, ["A.lean"], label="first")
        assert passed, log
        assert "root.compared-a1|A.lean" in log
        assert path.name == "comparator-v1-first.log"
        slow = runtime_for(
            loaded,
            project,
            tmp_path,
            comparator_command="bash tools/slow.sh",
            comparator_timeout=1,
        )
        passed, _, log = await slow._compare(node, [])
        assert not passed
        assert "comparator execution failed" in log

    asyncio.run(scenario())
    with pytest.raises(EnvCommandTimeout):
        asyncio.run(DirEnv(project, tmp_path).exec(["sleep", "30"], timeout=0.2))


def test_comparator_commands_receive_stable_candidate_base_and_no_token(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = repository(tmp_path / "comparator_base_problem")
    (project / "tools").mkdir()
    (project / "tools" / "check.sh").write_text(
        'printf "base=%s token=%s problem=%s\\n" "$HUMANIZE_CANDIDATE_BASE_COMMIT" '
        '"${HF_TOKEN-unset}" "$HUMANIZE_PROBLEM_MARKDOWN"\n'
        'printf "%s\\n" "Your solution is okay!"\n'
    )
    monkeypatch.setenv("HF_TOKEN", "bootstrap-only-secret")
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "comparator base fixture",
        lean_target="Submission.lean",
        comparator_command="bash tools/check.sh {node_id}",
    )
    node = loaded.models.NodeRecord(
        id="root.stable-base-a1",
        title="Stable base",
        statement="True",
        proof_base_commit="abc123",
    )

    rendered = runtime._review_command(node, ["Submission.lean"])

    assert rendered.startswith("env -u HF_TOKEN ")
    assert "HUMANIZE_CANDIDATE_BASE_COMMIT=abc123" in rendered
    assert f"HUMANIZE_PROBLEM_MARKDOWN={runtime.problem_path}" in rendered
    assert "tools/check.sh root.stable-base-a1" in rendered
    reviewed = subprocess.run(
        rendered, shell=True, cwd=project, capture_output=True, text=True, check=True
    )
    assert "base=abc123 token=unset" in reviewed.stdout
    passed, _, log = asyncio.run(runtime._compare(node, ["Submission.lean"]))
    assert passed, log
    assert f"base=abc123 token=unset problem={runtime.problem_path}" in log
    assert "bootstrap-only-secret" not in log


def test_agent_hidden_file_is_absent_from_primary_and_node_worktrees(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(tmp_path / "protected_problem")
    (project / "Solution.lean").write_text("theorem protected : True := by trivial\n")
    git(project, "add", "Solution.lean")
    git(project, "commit", "-m", "test: add the comparator-only solution")
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "fixture theorem",
        agent_hidden_files=["Solution.lean"],
    )
    node = loaded.models.NodeRecord(
        id="root.leaf-a1", title="Leaf", statement="True", attempts=1
    )

    async def scenario() -> Path:
        await runtime._seal_agent_workspace(runtime.workspace)
        assert await runtime._git_clean(runtime.workspace)
        worktree = await runtime._node_worktree(node)
        assert await runtime._git_clean(worktree)
        return Path(str(worktree.workdir))

    path = asyncio.run(scenario())

    for checkout in (project, path):
        assert not (checkout / "Solution.lean").exists()
        assert git(checkout, "ls-files", "-v", "Solution.lean").startswith("S ")
    context = runtime._problem_context()
    assert "`Solution.lean` are deliberately absent" in context
    assert "Do not recover or inspect them through Git objects/history" in context


def test_resealing_preserves_preexisting_participant_changes(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(tmp_path / "dirty_participant_worktree")
    (project / "Solution.lean").write_text("theorem protected : True := by trivial\n")
    git(project, "add", "Solution.lean")
    git(project, "commit", "-m", "test: add the comparator-only solution")
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "fixture theorem",
        agent_hidden_files=["Solution.lean"],
    )
    (project / "Submission.lean").write_text(
        "namespace Submission\ntheorem work : True := by trivial\nend Submission\n"
    )
    (project / "Submission").mkdir()
    (project / "Submission" / "Work.lean").write_text(
        "namespace Submission\ntheorem moreWork : True := by trivial\nend Submission\n"
    )

    async def scenario() -> tuple[str, str]:
        before = await runtime._git_status_snapshot(runtime.workspace)
        await runtime._seal_agent_workspace(runtime.workspace)
        await runtime._seal_agent_workspace(runtime.workspace)
        return before, await runtime._git_status_snapshot(runtime.workspace)

    before, after = asyncio.run(scenario())

    assert after == before
    assert not (project / "Solution.lean").exists()
    assert git(project, "ls-files", "-v", "Solution.lean").startswith("S ")


def test_long_node_identities_do_not_collide(loaded: Any, tmp_path: Path) -> None:
    project = tmp_path / "identity_problem"
    project.mkdir()
    runtime = runtime_for(loaded, project, tmp_path, "identity fixture")
    prefix = "root." + "shared_collision_segment_" * 5
    left, right, legacy_node = (
        loaded.models.NodeRecord(
            id=prefix + suffix, title=suffix, statement="True", attempts=1
        )
        for suffix in ("nontrivial-a1", "torsion-a1", "legacy-a1")
    )
    slug = loaded.store.slug

    assert slug(left.id) == slug(right.id)
    assert runtime._node_branch(left) != runtime._node_branch(right)
    assert runtime._node_dir(left) != runtime._node_dir(right)
    assert len(runtime._node_dir(left).name) <= 80
    assert len(runtime._node_dir(right).name) <= 80
    worktrees = tmp_path / "worktrees"
    assert runtime._node_worktree_path(left, worktrees) != (
        runtime._node_worktree_path(right, worktrees)
    )

    # A run made before identities were told apart keeps the directory it wrote.
    legacy = runtime.run_root / "nodes" / slug(legacy_node.id)
    legacy.mkdir(parents=True)
    assert runtime._node_dir(legacy_node) == legacy


# ---------------------------------------------------------------------------- integration


def test_accepted_candidate_retries_only_integration(
    loaded: Any, tmp_path: Path
) -> None:
    project = tmp_path / "integration_retry_problem"
    project.mkdir()
    runtime = runtime_for(loaded, project, tmp_path, "integration retry fixture")
    node = runtime.store.ensure(
        "root.accepted-a1",
        parent="root",
        depth=1,
        title="Accepted theorem",
        statement="True",
    )
    node.status = "integrating"
    node.natural_proof = "accepted-natural-proof.md"
    node.worktree = "/tmp/accepted-proof-worktree"
    node.proof_branch = "humanize-recursive/accepted"
    node.proof_base_commit = "base"
    node.candidate_commit = "candidate"
    pause = AsyncMock()

    with (
        patch.object(
            runtime,
            "_integrate_candidate",
            side_effect=[
                (False, "combined history failed"),
                (True, "agent-reconciled and integrated"),
            ],
        ) as integrate,
        patch.object(loaded.runtime.asyncio, "sleep", pause),
    ):
        accepted, feedback = asyncio.run(
            runtime._integrate_reviewed_candidate(
                "base", "candidate", node=node, lean_files=["Submission.lean"]
            )
        )

    assert accepted
    assert feedback == "agent-reconciled and integrated"
    assert integrate.call_count == 2
    pause.assert_awaited_once_with(1.0)
    record = runtime.store.nodes[node.id]
    assert record.status == "integrating"
    assert "accepted proof retained" in record.message
    assert record.natural_proof == "accepted-natural-proof.md"
    assert record.worktree == "/tmp/accepted-proof-worktree"
    assert record.proof_branch == "humanize-recursive/accepted"
    assert record.proof_base_commit == "base"
    assert record.candidate_commit == "candidate"


def _leaves(loaded: Any, names: tuple[str, ...]) -> list[Any]:
    return [
        loaded.models.NodeRecord(
            id=f"root.{name}-a1", title=f"Leaf {name}", statement="True", attempts=1
        )
        for name in names
    ]


def test_two_ready_leaf_histories_integrate_from_parallel_worktrees(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(tmp_path / "parallel_problem")
    runtime = runtime_for(loaded, project, tmp_path, "parallel fixture")

    async def scenario() -> list[tuple[bool, str]]:
        worktrees = [
            await runtime._node_worktree(node)
            for node in _leaves(loaded, ("leaf_1", "leaf_2"))
        ]
        bases = [await runtime._git_head(worktree) for worktree in worktrees]
        heads: list[str] = []
        for number, worktree in enumerate(worktrees, 1):
            path = Path(str(worktree.workdir))
            (path / f"Leaf{number}.lean").write_text(
                f"theorem leaf{number} : True := by trivial\n"
            )
            git(path, "add", f"Leaf{number}.lean")
            git(path, "commit", "-m", f"feat: prove leaf {number}")
            heads.append(await runtime._git_head(worktree))
        return await asyncio.gather(
            *(
                runtime._integrate_candidate(before, after)
                for before, after in zip(bases, heads, strict=True)
            )
        )

    results = asyncio.run(scenario())

    assert all(passed for passed, _ in results), results
    assert (project / "Leaf1.lean").is_file()
    assert (project / "Leaf2.lean").is_file()
    assert git(project, "status", "--porcelain") == ""


def test_divergent_integration_supplies_its_own_committer_identity(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = repository(
        tmp_path / "identity_problem",
        submission="namespace Submission\n\nend Submission\n",
        author="Fixture Author",
    )
    runtime = runtime_for(loaded, project, tmp_path, "identity fixture")
    [node] = _leaves(loaded, ("identity_leaf",))

    async def candidate() -> tuple[str, str]:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        before = await runtime._git_head(worktree)
        (path / "Candidate.lean").write_text("theorem candidate : True := by trivial\n")
        git(path, "add", "Candidate.lean")
        git(path, "commit", "-m", "feat: add candidate")
        return before, await runtime._git_head(worktree)

    before, after = asyncio.run(candidate())
    (project / "Canonical.lean").write_text("theorem canonical : True := by trivial\n")
    git(project, "add", "Canonical.lean")
    git(project, "commit", "-m", "feat: advance canonical")
    git(project, "config", "--unset-all", "user.name")
    git(project, "config", "--unset-all", "user.email")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    integrated, feedback = asyncio.run(runtime._integrate_candidate(before, after))

    assert integrated, feedback
    assert "rebased" in feedback
    assert (project / "Candidate.lean").is_file()
    assert (project / "Canonical.lean").is_file()


def test_stale_proof_base_skips_commits_already_on_canonical(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(
        tmp_path / "stale_base_problem",
        submission="namespace Submission\n\nend Submission\n",
    )
    runtime = runtime_for(loaded, project, tmp_path, "stale-base fixture")
    [node] = _leaves(loaded, ("stale_base",))

    async def scenario() -> tuple[bool, str]:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        stale_before = await runtime._git_head(worktree)
        (project / "Canonical.lean").write_text(
            "theorem canonical : True := by trivial\n"
        )
        git(project, "add", "Canonical.lean")
        git(project, "commit", "-m", "feat: advance canonical")
        canonical = await runtime._git_head()
        git(path, "merge", "--ff-only", canonical)
        (path / "Candidate.lean").write_text("theorem candidate : True := by trivial\n")
        git(path, "add", "Candidate.lean")
        git(path, "commit", "-m", "feat: prove candidate")
        after = await runtime._git_head(worktree)
        return await runtime._integrate_candidate(stale_before, after)

    integrated, feedback = asyncio.run(scenario())

    assert integrated, feedback
    assert (project / "Canonical.lean").is_file()
    assert (project / "Candidate.lean").is_file()


def test_duplicate_patch_is_an_accepted_empty_cherry_pick(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(
        tmp_path / "duplicate_patch_problem",
        submission="namespace Submission\n\nend Submission\n",
    )
    runtime = runtime_for(loaded, project, tmp_path, "duplicate-patch fixture")
    [node] = _leaves(loaded, ("duplicate",))
    duplicate = "theorem duplicate : True := by trivial\n"

    async def scenario() -> tuple[bool, str]:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        before = await runtime._git_head(worktree)
        (path / "Duplicate.lean").write_text(duplicate)
        git(path, "add", "Duplicate.lean")
        git(path, "commit", "-m", "feat: candidate copy")
        after = await runtime._git_head(worktree)
        (project / "Duplicate.lean").write_text(duplicate)
        git(project, "add", "Duplicate.lean")
        git(project, "commit", "-m", "feat: canonical copy")
        return await runtime._integrate_candidate(before, after)

    integrated, feedback = asyncio.run(scenario())

    assert integrated, feedback
    assert (project / "Duplicate.lean").read_text() == duplicate
    assert git(project, "status", "--porcelain") == ""


def test_parallel_same_file_leaf_additions_are_union_integrated(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A byte that is not UTF-8 in the reconciled file comes through the union untouched.
    project = repository(
        tmp_path / "same_file_problem",
        submission="namespace Submission\n\nend Submission\n",
    )
    (project / "Submission.lean").write_bytes(
        b"namespace Submission\n-- caf\xe9\n\nend Submission\n"
    )
    git(project, "commit", "-am", "test: a Latin-1 comment")
    runtime = runtime_for(loaded, project, tmp_path, "same-file fixture")

    async def candidates() -> list[tuple[str, str]]:
        made: list[tuple[str, str]] = []
        for number, node in enumerate(_leaves(loaded, ("same_1", "same_2")), 1):
            worktree = await runtime._node_worktree(node)
            path = Path(str(worktree.workdir))
            before = await runtime._git_head(worktree)
            (path / "Submission.lean").write_bytes(
                b"namespace Submission\n-- caf\xe9\n\n"
                + f"theorem same{number} : True := by trivial\n\n".encode()
                + b"end Submission\n"
            )
            git(path, "add", "Submission.lean")
            git(path, "commit", "-m", f"feat: prove same-file leaf {number}")
            made.append((before, await runtime._git_head(worktree)))
        return made

    histories = asyncio.run(candidates())
    git(project, "config", "--unset-all", "user.name")
    git(project, "config", "--unset-all", "user.email")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    async def integrate() -> list[tuple[bool, str]]:
        return [await runtime._integrate_candidate(*one) for one in histories]

    first, second = asyncio.run(integrate())

    assert first[0], first
    assert second[0], second
    assert "union-reconciled" in second[1]
    combined = (project / "Submission.lean").read_bytes()
    assert b"theorem same1" in combined
    assert b"theorem same2" in combined
    assert combined.count(b"-- caf\xe9\n") == 1
    assert git(project, "status", "--porcelain") == ""


def test_empty_cherry_pick_after_lean_union_is_accepted(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(
        tmp_path / "empty_union_problem", submission="theorem value : Nat := 0\n"
    )
    runtime = runtime_for(loaded, project, tmp_path, "empty-union fixture")
    [node] = _leaves(loaded, ("union_duplicate",))

    async def keep_canonical(integration: Any) -> tuple[bool, str]:
        await integration.exec(["git", "checkout", "--ours", "Submission.lean"])
        await integration.exec(["git", "add", "Submission.lean"])
        return True, "kept already-integrated canonical Lean source"

    async def scenario() -> tuple[bool, bool, str]:
        worktree = await runtime._node_worktree(node)
        path = Path(str(worktree.workdir))
        (path / "Submission.lean").write_text("theorem value : Nat := 1\n")
        git(path, "add", "Submission.lean")
        git(path, "commit", "-m", "feat: candidate value")
        candidate = await runtime._git_head(worktree)
        (project / "Submission.lean").write_text("theorem value : Nat := 2\n")
        git(project, "add", "Submission.lean")
        git(project, "commit", "-m", "feat: canonical value")
        with patch.object(runtime, "_union_lean_conflicts", keep_canonical):
            return await runtime._apply_candidate_commits(
                runtime.workspace, [candidate]
            )

    applied, unioned, feedback = asyncio.run(scenario())

    assert applied, feedback
    assert unioned
    assert (project / "Submission.lean").read_text() == "theorem value : Nat := 2\n"
    assert git(project, "status", "--porcelain") == ""


def test_parent_worktree_overlays_accepted_child_candidate(
    loaded: Any, tmp_path: Path
) -> None:
    project = repository(tmp_path / "accepted_overlay_problem")
    runtime = runtime_for(
        loaded, project, tmp_path, "accepted overlay fixture", max_parallel_children=4
    )
    parent = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="Root theorem"
    )
    child = runtime.store.ensure(
        "root.child-a1",
        parent="root",
        depth=1,
        title="Child",
        statement="Child theorem",
        lean_name="accepted_child",
    )
    child.attempts = 1

    async def scenario() -> tuple[bool, str, Path]:
        child_worktree = await runtime._node_worktree(child)
        child_path = Path(str(child_worktree.workdir))
        child_base = child.proof_base_commit
        (child_path / "Child.lean").write_text(
            "theorem accepted_child : True := by trivial\n"
        )
        git(child_path, "add", "Child.lean")
        git(child_path, "commit", "-m", "feat: prove accepted child")
        child.status = "integrating"
        child.candidate_commit = await runtime._git_head(child_worktree)
        child.theorems = ["Submission.accepted_child"]
        assert child.proof_base_commit == child_base

        parent.attempts = 1
        parent_worktree = await runtime._node_worktree(parent)
        passed, feedback = await runtime._overlay_accepted_children(
            parent, parent_worktree
        )
        return passed, feedback, Path(str(parent_worktree.workdir))

    passed, feedback, parent_path = asyncio.run(scenario())

    assert passed, feedback
    assert (parent_path / "Child.lean").read_text() == (
        "theorem accepted_child : True := by trivial\n"
    )


# ----------------------------------------------------------------- the natural-language gate


def _scripted(runtime: Any, **replies: list[Any]) -> list[tuple[str, str]]:
    """Answers each role's turns, in order, from `replies`; and every turn asked of it."""
    runtime.worker = SimpleNamespace(role="worker")
    runtime.reviewer = SimpleNamespace(role="reviewer")
    queued = {role: iter(answers) for role, answers in replies.items()}
    asked: list[tuple[str, str]] = []

    async def ask(agent: Any, prompt: str, schema: Any = None, env: Any = None) -> Any:
        asked.append((agent.role, prompt))
        return next(queued[agent.role])

    runtime._ask = ask
    return asked


def _natural_proof(loaded: Any) -> Any:
    return loaded.models.NaturalProof(
        reference_use=REFERENCE_USE,
        proof=(
            "1. Establish the required intermediate lemma.\n"
            "2. Apply it to prove the exact frozen theorem."
        ),
        key_steps=["Establish and apply the intermediate lemma."],
        unresolved=[],
    )


def _natural_runtime(
    loaded: Any, tmp_path: Path, name: str, **params: Any
) -> tuple[Any, Path]:
    project = tmp_path / name
    project.mkdir()
    runtime = runtime_for(loaded, project, tmp_path, **params)
    runtime.reference_bundle = reference_bundle(loaded, tmp_path / "references")
    plan = project / "plan.md"
    plan.write_text("# Accepted plan\n")
    return runtime, plan


def test_child_contract_contradiction_returns_to_parent(
    loaded: Any, tmp_path: Path
) -> None:
    runtime, plan = _natural_runtime(
        loaded, tmp_path, "contradicted_child_problem", natural_proof_attempts=1
    )
    audit = loaded.models.NaturalAudit(
        reference_use=REFERENCE_USE,
        acceptable=False,
        first_invalid_step="Before Step 1: the exact frozen child proposition is false.",
        required_changes=["Revise the parent decomposition."],
        requires_parent_revision=True,
        contract_contradiction=(
            "For the explicit fixture x, the frozen conclusion says x = 0, "
            "while direct evaluation gives x = 1."
        ),
    )
    asked = _scripted(runtime, worker=[_natural_proof(loaded)], reviewer=[audit])
    node = runtime.store.ensure(
        "root.false_child-a1",
        parent="root",
        depth=1,
        title="False child",
        statement="Every fixture is zero",
        lean_name="false_child",
        lean_statement="∀ x : Nat, x = 0",
    )

    accepted = asyncio.run(runtime._accepted_natural_proof(node, plan))

    assert accepted is None
    assert node.status == "failed"
    assert "frozen child contract" in node.message
    assert "direct evaluation gives x = 1" in node.message
    assert [role for role, _ in asked] == ["worker", "reviewer"]
    review = asked[1][1]
    assert "- DAG node: `root.false_child-a1`" in review
    assert "- Lean declaration: `Submission.false_child`" in review
    assert "- Frozen Lean proposition: `∀ x : Nat, x = 0`" in review


def test_failed_child_natural_gate_unwinds_solve(loaded: Any, tmp_path: Path) -> None:
    runtime, plan = _natural_runtime(loaded, tmp_path, "parent_backtrack_problem")
    node = runtime.store.ensure(
        "root.false_child-a1",
        parent="root",
        depth=1,
        title="False child",
        statement="False child statement",
    )
    node.plan = plan.name

    async def reject_contract(*_: Any) -> None:
        runtime.store.update(
            node.id, "failed", "reviewer-certified child contradiction"
        )

    with (
        patch.object(runtime, "_accepted_natural_proof", side_effect=reject_contract),
        patch.object(runtime, "_revise_parent") as revise_parent,
    ):
        result = asyncio.run(runtime._solve(node))

    assert not result.ok
    assert result.node_id == node.id
    assert "child contradiction" in result.feedback
    revise_parent.assert_called_once_with(node, node.message)


def test_parent_revision_certificate_requires_concrete_contradiction(
    loaded: Any,
) -> None:
    fixture: dict[str, Any] = {
        "reference_use": REFERENCE_USE,
        "acceptable": False,
        "first_invalid_step": "The child target is false.",
        "required_changes": ["Revise the parent decomposition."],
        "requires_parent_revision": True,
    }
    audit = loaded.models.NaturalAudit
    with pytest.raises(pydantic.ValidationError, match="concrete contract"):
        audit(**fixture)
    with pytest.raises(pydantic.ValidationError, match="cannot be acceptable"):
        audit(
            **fixture
            | {"acceptable": True, "contract_contradiction": "A contradiction."}
        )
    with pytest.raises(pydantic.ValidationError, match="requires_parent_revision"):
        audit(
            **fixture
            | {
                "requires_parent_revision": False,
                "contract_contradiction": "A contradiction.",
            }
        )


def test_null_natural_audit_retries_same_draft_before_reauthoring(
    loaded: Any, tmp_path: Path
) -> None:
    runtime, plan = _natural_runtime(
        loaded, tmp_path, "natural_review_retry_problem", natural_proof_attempts=2
    )
    proof = _natural_proof(loaded)
    audit = loaded.models.NaturalAudit(
        reference_use=REFERENCE_USE,
        acceptable=True,
        first_invalid_step="",
        required_changes=[],
    )
    asked = _scripted(runtime, worker=[proof], reviewer=[None, audit])
    node = runtime.store.ensure(
        "root",
        parent=None,
        depth=0,
        title="Retry reviewer",
        statement="True",
        lean_name="retry_reviewer",
        lean_statement="True",
    )
    pause = AsyncMock()

    with patch.object(loaded.runtime.asyncio, "sleep", pause):
        accepted = asyncio.run(runtime._accepted_natural_proof(node, plan))

    assert accepted is proof
    reviews = [prompt for role, prompt in asked if role == "reviewer"]
    assert [role for role, _ in asked] == ["worker", "reviewer", "reviewer"]
    assert "previous reviewer call" not in reviews[0]
    assert "previous reviewer call" in reviews[1]
    pause.assert_awaited_once_with(15.0)
    assert node.status == "decomposing"
    node_dir = runtime._node_dir(node)
    assert (node_dir / "natural-review-null-v1-attempt-1.txt").is_file()
    assert (node_dir / "natural-audit-v1.json").is_file()
    accepted_proof = (runtime.project / node.natural_proof).read_text()
    assert "## Reference use\n\n### TauCeti" in accepted_proof


def test_null_natural_audit_uses_json_transport_fallback(
    loaded: Any, tmp_path: Path
) -> None:
    runtime, plan = _natural_runtime(
        loaded,
        tmp_path,
        "natural_review_json_fallback_problem",
        natural_proof_attempts=2,
    )
    proof = _natural_proof(loaded)
    audit = loaded.models.NaturalAudit(
        reference_use=REFERENCE_USE,
        acceptable=True,
        first_invalid_step="",
        required_changes=[],
    )
    asked = _scripted(
        runtime,
        worker=[proof],
        reviewer=[None, None, f"```json\n{audit.model_dump_json()}\n```"],
    )
    node = runtime.store.ensure(
        "root",
        parent=None,
        depth=0,
        title="Recover reviewer JSON",
        statement="True",
        lean_name="recover_reviewer_json",
        lean_statement="True",
    )
    pause = AsyncMock()

    with patch.object(loaded.runtime.asyncio, "sleep", pause):
        accepted = asyncio.run(runtime._accepted_natural_proof(node, plan))

    assert accepted is proof
    assert [role for role, _ in asked] == ["worker", *["reviewer"] * 3]
    assert "JSON Schema" in asked[-1][1]
    pause.assert_awaited_once_with(15.0)
    assert node.status == "decomposing"
    node_dir = runtime._node_dir(node)
    assert (node_dir / "natural-audit-v1.json").is_file()
    marker = node_dir / "natural-review-json-fallback-v1.txt"
    assert "produced a valid" in marker.read_text()


def test_reference_evidence_outside_the_snapshots_is_fed_back(
    loaded: Any, tmp_path: Path
) -> None:
    runtime, plan = _natural_runtime(
        loaded, tmp_path, "reference_problem", natural_proof_attempts=1
    )
    stray = _natural_proof(loaded).model_dump()
    stray["reference_use"][0]["files"] = [str(tmp_path / "elsewhere.md")]
    (tmp_path / "elsewhere.md").write_text("not a snapshot\n")
    audit = loaded.models.NaturalAudit(
        reference_use=REFERENCE_USE,
        acceptable=True,
        first_invalid_step="",
        required_changes=[],
    )
    asked = _scripted(
        runtime,
        worker=[loaded.models.NaturalProof(**stray), _natural_proof(loaded)],
        reviewer=[audit],
    )
    node = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="True"
    )

    accepted = asyncio.run(runtime._accepted_natural_proof(node, plan))

    assert accepted is not None
    assert [role for role, _ in asked] == ["worker", "worker", "reviewer"]
    assert "did not cite an existing path" in asked[1][1]


# ------------------------------------------------------------------------------ the frontier


def _subproblem(loaded: Any, key: str, *depends_on: str, name: str = "") -> Any:
    return loaded.models.Subproblem(
        key=key,
        title=f"Lemma {key}",
        statement=f"An independent theorem called {key}",
        lean_statement="True",
        lean_name=name or key,
        depends_on=list(depends_on),
    )


def test_child_frontier_refills_without_waiting_for_slow_sibling(
    loaded: Any, tmp_path: Path
) -> None:
    project = tmp_path / "frontier_problem"
    project.mkdir()
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "frontier fixture",
        max_nodes=10,
        max_parallel_children=4,
    )
    parent = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="True"
    )
    decomposition = loaded.models.Decomposition(
        reference_use=REFERENCE_USE,
        should_split=True,
        rationale="three-node dependency fixture",
        subproblems=[
            _subproblem(loaded, "slow"),
            _subproblem(loaded, "fast"),
            _subproblem(loaded, "after_fast", "fast"),
        ],
    )
    moments: dict[str, float] = {}

    async def solve(node: Any) -> Any:
        key = node.id.rsplit(".", 1)[-1].rsplit("-a", 1)[0]
        loop = asyncio.get_running_loop()
        moments[f"start:{key}"] = loop.time()
        await asyncio.sleep(0.25 if key == "slow" else 0.02)
        moments[f"end:{key}"] = loop.time()
        return loaded.models.SolveResult(ok=True, node_id=node.id)

    runtime._solve = solve
    results = asyncio.run(runtime._solve_children(parent, decomposition))

    assert all(result.ok for result in results)
    assert moments["start:after_fast"] < moments["end:slow"]


def test_redecomposition_reuses_proved_theorem_instead_of_creating_a2(
    loaded: Any, tmp_path: Path
) -> None:
    project = tmp_path / "reuse_problem"
    project.mkdir()
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "reuse fixture",
        max_nodes=10,
        max_parallel_children=4,
    )
    parent = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="Root theorem"
    )
    child = runtime.store.ensure(
        "root.lemma-a1",
        parent="root",
        depth=1,
        title="Lemma",
        statement="Original statement",
        lean_statement="True",
        lean_name="stable_lemma",
    )
    child.status = "proved"
    child.theorems = ["Submission.stable_lemma"]
    decomposition = loaded.models.Decomposition(
        reference_use=REFERENCE_USE,
        should_split=True,
        rationale="retry with the same theorem identity",
        subproblems=[
            _subproblem(loaded, "renamed_key", name="stable_lemma"),
            _subproblem(loaded, "second", name="second_lemma"),
        ],
    )
    seen: list[str] = []

    async def solve(node: Any) -> Any:
        seen.append(node.id)
        node.status = "proved"
        node.theorems = [f"Submission.{node.lean_name}"]
        return loaded.models.SolveResult(
            ok=True, node_id=node.id, theorems=runtime._checkpoint_theorems(node)
        )

    runtime._solve = solve
    results = asyncio.run(runtime._solve_children(parent, decomposition))

    assert all(result.ok for result in results)
    assert "root.lemma-a1" not in seen
    assert "root.second-a1" in seen
    assert parent.children == ["root.lemma-a1", "root.second-a1"]
    assert not any(node_id.endswith("-a2") for node_id in runtime.store.nodes)


def test_integrating_child_unlocks_its_dependent_without_reproving(
    loaded: Any, tmp_path: Path
) -> None:
    project = tmp_path / "accepted_frontier_problem"
    project.mkdir()
    runtime = runtime_for(
        loaded,
        project,
        tmp_path,
        "accepted frontier fixture",
        max_nodes=10,
        max_parallel_children=4,
    )
    parent = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="Root theorem"
    )
    accepted = runtime.store.ensure(
        "root.accepted-a1",
        parent="root",
        depth=1,
        title="Accepted child",
        statement="An accepted child theorem",
        lean_statement="True",
        lean_name="accepted_child",
    )
    accepted.status = "integrating"
    accepted.candidate_commit = "candidate"
    accepted.theorems = ["Submission.accepted_child"]
    decomposition = loaded.models.Decomposition(
        reference_use=REFERENCE_USE,
        should_split=True,
        rationale="one accepted prerequisite and its dependent",
        subproblems=[
            _subproblem(loaded, "accepted", name="accepted_child"),
            _subproblem(loaded, "dependent", "accepted", name="dependent_child"),
        ],
    )
    started: list[str] = []

    async def solve(node: Any) -> Any:
        started.append(node.id)
        node.status = "proved"
        node.theorems = [f"Submission.{node.lean_name}"]
        return loaded.models.SolveResult(
            ok=True, node_id=node.id, theorems=runtime._checkpoint_theorems(node)
        )

    runtime._solve = solve
    with patch.object(runtime, "_integrate_later") as promote:
        results = asyncio.run(runtime._solve_children(parent, decomposition))

    assert all(result.ok for result in results)
    promote.assert_called_once_with(accepted)
    assert started == ["root.dependent-a1"]


def test_a_spent_budget_in_one_child_stops_the_frontier(
    loaded: Any, tmp_path: Path
) -> None:
    from hmz.flows import CostExceeded

    project = tmp_path / "budget_problem"
    project.mkdir()
    runtime = runtime_for(loaded, project, tmp_path, "budget fixture", max_nodes=10)
    parent = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="Root theorem"
    )
    decomposition = loaded.models.Decomposition(
        reference_use=REFERENCE_USE,
        should_split=True,
        rationale="two independent lemmas",
        subproblems=[_subproblem(loaded, "spent"), _subproblem(loaded, "slow")],
    )
    cancelled: list[str] = []

    async def solve(node: Any) -> Any:
        if "spent" in node.id:
            raise CostExceeded("the run's cost is spent")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(node.id)
            raise
        return loaded.models.SolveResult(ok=True, node_id=node.id)

    runtime._solve = solve
    with pytest.raises(CostExceeded):
        asyncio.run(runtime._solve_children(parent, decomposition))
    assert cancelled == ["root.slow-a1"]
    assert runtime.store.nodes["root.spent-a1"].status != "failed"


# ---------------------------------------------------------------------------- a whole run


FORMALIZE = re.compile(r"Formalize DAG node `([^`]+)` only")
REVIEWED = re.compile(r"^Node: (\S+)$", re.MULTILINE)


def _lean_name(node_id: str) -> str:
    return node_id.rsplit(".", 1)[-1].rsplit("-a", 1)[0]


def _lean_file(node_id: str) -> str:
    return f"{_lean_name(node_id).capitalize()}.lean"


def _agents(
    loaded: Any, *, interrupt: str = ""
) -> tuple[FakeAgentDriver, FakeAgentDriver, list[Any]]:
    """A worker and a reviewer that prove anything, and where each of their turns was.

    The reviewer raises, as a run that was killed would stop, on reviewing the Lean of the
    node `interrupt` names.
    """
    models = loaded.models
    turns: list[tuple[str, Any, Path, str]] = []
    # How many of an agent's sessions were open as each of its turns was taken.
    opened: list[int] = []

    def worker(prompt: str, *, output_schema: Any, session: Any) -> Any:
        where = Path(str(session.placement.workdir))
        turns.append(("worker", output_schema, where, prompt))
        opened.append(sum(not one.closed for one in session.driver.sessions))
        if output_schema is models.FetchedProblem:
            return fetched_problem(loaded, where.name)
        if output_schema is models.NaturalProof:
            return models.NaturalProof(
                reference_use=REFERENCE_USE,
                proof="1. Every step follows from the definitions, so the theorem holds.",
                key_steps=["unfold the definitions"],
                unresolved=[],
            )
        if output_schema is models.Decomposition:
            return models.Decomposition(
                reference_use=REFERENCE_USE,
                should_split=True,
                rationale="two reusable lemmas",
                subproblems=[_subproblem(loaded, "alpha"), _subproblem(loaded, "beta")],
            )
        formalized = FORMALIZE.search(prompt)
        if formalized is None:
            return "# Scaffold\n\n1. Prove the lemmas, then the theorem.\n"
        node_id = formalized.group(1)
        (where / _lean_file(node_id)).write_text(
            f"theorem {_lean_name(node_id)} : True := by trivial\n"
        )
        if git(where, "status", "--porcelain"):
            git(where, "add", _lean_file(node_id))
            git(where, "commit", "-m", f"feat: prove {_lean_name(node_id)}")
        return "implemented and committed"

    def reviewer(prompt: str, *, output_schema: Any, session: Any) -> Any:
        where = Path(str(session.placement.workdir))
        turns.append(("reviewer", output_schema, where, prompt))
        opened.append(sum(not one.closed for one in session.driver.sessions))
        if output_schema is models.NaturalAudit:
            return models.NaturalAudit(
                reference_use=REFERENCE_USE,
                acceptable=True,
                first_invalid_step="",
                required_changes=[],
            )
        if output_schema is models.DecompositionAudit:
            return models.DecompositionAudit(
                reference_use=REFERENCE_USE,
                acceptable=True,
                nodes=[
                    models.SubproblemAudit(key=key, acceptable=True, reason="sound")
                    for key in ("alpha", "beta")
                ],
                required_changes=[],
            )
        if output_schema is models.LeanAudit:
            reviewed = REVIEWED.search(prompt)
            assert reviewed is not None
            node_id = reviewed.group(1)
            if node_id == interrupt:
                raise RuntimeError("interrupted")
            return models.LeanAudit(
                reference_use=REFERENCE_USE,
                accepted=True,
                comparator_reran=True,
                comparator_passed=True,
                proof_matches_statement=True,
                issues=[],
                theorems=[
                    models.ProvedTheorem(
                        name=f"Submission.{_lean_name(node_id)}",
                        statement="True",
                        lean_file=_lean_file(node_id),
                        natural_summary="It holds by unfolding the definitions.",
                    )
                ],
            )
        return "ok"

    working = FakeAgentDriver(HarnessKind.CODEX, reply=worker, model="worker-model")
    working.opened = opened  # type: ignore[attr-defined]
    return (
        working,
        FakeAgentDriver(HarnessKind.CODEX, reply=reviewer, model="reviewer-model"),
        turns,
    )


class Recorder:
    """Every flow call a run made, as the engine tells a recorder of them."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def entered(self, call: Any) -> None:
        self.calls.append(call)

    def left(self, call: Any, error: BaseException | None) -> None:
        del call, error

    def spawned(self, call: Any, role: str, session: Any, driver: Any) -> None:
        del call, role, session, driver


def _problem(tmp_path: Path) -> Path:
    project = repository(tmp_path / "problem")
    (project / "tools").mkdir()
    (project / "tools" / "check-with-comparator.sh").write_text(COMPARATOR)
    git(project, "add", "tools")
    git(project, "commit", "-m", "test: add the comparator")
    return project


def _run_dir(project: Path) -> Path:
    latest = project / ".humanize" / "recursive-lean-prover" / "LATEST"
    return project / latest.read_text().strip()


def _dag(project: Path) -> dict[str, Any]:
    held = json.loads((_run_dir(project) / "dag.json").read_text())
    return {one["id"]: one for one in held["nodes"]}


def test_a_root_splits_into_two_children_that_are_proved_integrated_and_published(
    loaded: Any, tmp_path: Path, offline: list[Any]
) -> None:
    project = _problem(tmp_path)
    worker, reviewer, turns = _agents(loaded)
    recorder = Recorder()
    loaded.calls.clear()
    scratch = tmp_path / "scratch"
    workspace = DirEnv(project, scratch)

    asyncio.run(
        run_fake(
            loaded.entry,
            "Prove that the main theorem holds.",
            agents={"worker": worker, "reviewer": reviewer},
            params={"max_depth": "1", "max_nodes": "3", "rlcr_rounds": "5"},
            local=workspace,
            recorder=recorder,
        )
    )

    # Preflight came first: the snapshots, then the one problem, frozen in the run.
    run_dir = _run_dir(project)
    [bundle] = offline
    assert bundle.root == project / ".humanize" / "math-reference-library"
    assert json.loads((run_dir / "preflight.json").read_text())["status"] == "ready"
    problem = (run_dir / "problem.md").read_text()
    assert problem.startswith("# Mihăilescu's theorem\n")
    assert "| Problem id | `problem` |" in problem
    assert json.loads((run_dir / "problem.json").read_text())["problem_id"] == "problem"
    fetches = [one for one in turns if one[1] is loaded.models.FetchedProblem]
    assert len(fetches) == 1
    assert "only permitted problem id: `problem`" in fetches[0][3]
    reasoning = [prompt for _, schema, _, prompt in turns if schema is not None]
    assert all(str(run_dir / "problem.md") in one for one in reasoning[1:])
    assert all(str(bundle.manifest) in one for one in reasoning[1:])

    # The DAG: a root and its two children, every one proved.
    nodes = _dag(project)
    assert set(nodes) == {"root", "root.alpha-a1", "root.beta-a1"}
    assert {one["status"] for one in nodes.values()} == {"proved"}
    assert nodes["root"]["children"] == ["root.alpha-a1", "root.beta-a1"]
    rendered = (run_dir / "DAG.md").read_text()
    assert "n_root --> n_root_alpha_a1" in rendered
    assert "n_root --> n_root_beta_a1" in rendered
    assert rendered.count("| proved |") == 3
    assert f"Fetched problem: `{run_dir / 'problem.md'}`" in rendered
    assert f"Reference manifest: `{bundle.manifest}`" in rendered

    # Every accepted theorem is in the problem branch, which is clean, and in the wiki.
    for name in ("Alpha.lean", "Beta.lean", "Root.lean"):
        assert (project / name).is_file()
    assert git(project, "status", "--porcelain") == ""
    assert git(project, "branch", "--show-current") == "main"
    wiki = project / ".humanize" / "math-wiki"
    for name in ("alpha", "beta", "root"):
        page = (wiki / f"submission-{name}.md").read_text()
        assert f"# `Submission.{name}`" in page
        assert "Your solution is okay!" in page
    index = (wiki / "README.md").read_text()
    assert "[submission-alpha](submission-alpha.md)" in index
    compared = set((run_dir / "compared.txt").read_text().split())
    assert {"root", "root.alpha-a1", "root.beta-a1"} <= compared

    # humanize1's flows were called in-process, with this flow's agents and environments.
    plans = [one for one in loaded.calls if one["flow"] == "gen-plan"]
    builds = [one for one in loaded.calls if one["flow"] == "rlcr"]
    assert len(plans) == 3
    assert len(builds) == 3
    for plan in plans:
        assert plan["agents"] == {
            "planner": "worker-model",
            "analyst": "reviewer-model",
        }
        assert plan["workspace"] == str(project)
        assert plan["params"]["mode"] == "direct"
        assert plan["params"]["auto_start_rlcr_if_converged"] is False
        assert plan["params"]["turn_retries"] == 1
        assert plan["params"]["output"].endswith("/plan-v1.md")
        assert (project / plan["params"]["output"]).is_file()
    worktrees = scratch / "recursive-lean-worktrees"
    for build in builds:
        assert build["agents"] == {
            "builder": "worker-model",
            "reviewer": "reviewer-model",
        }
        assert Path(build["workspace"]).is_relative_to(worktrees)
        assert Path(build["workspace"]).name == project.name
        assert build["params"]["base_branch"] == ""
        assert build["params"]["skip_code_review"] is True
        assert build["params"]["max"] == 5
        assert build["params"]["plan_file"].endswith("/rlcr-plan-v1.md")
    assert len({build["workspace"] for build in builds}) == 3

    names = [(call.name, call.depth) for call in recorder.calls]
    assert names.count(("recursive_lean_prover", 1)) == 1
    assert names.count(("gen-plan", 2)) == 3
    assert names.count(("worktree-rlcr", 2)) == 3
    assert names.count(("rlcr", 3)) == 3

    # Each turn ran where its work is: prose in the problem, Lean in the node's worktree.
    models = loaded.models
    for role, schema, where, _ in turns:
        if schema in (models.NaturalProof, models.NaturalAudit, models.Decomposition):
            assert where == project, role
    audits = [where for role, schema, where, _ in turns if schema is models.LeanAudit]
    assert sorted(map(str, audits)) == sorted(build["workspace"] for build in builds)
    built = [where for role, schema, where, prompt in turns if FORMALIZE.search(prompt)]
    assert sorted(map(str, built)) == sorted(build["workspace"] for build in builds)
    # Its own sessions carry its skill and permission; humanize1's carry what that declares.
    sessions = [*worker.sessions, *reviewer.sessions]
    mine = [one for one in sessions if one.permission == loaded.flow.ROLE_PERMISSION]
    theirs = [one for one in sessions if one.permission == Permission()]
    assert len(mine) + len(theirs) == len(sessions)
    assert mine
    assert all(
        [skill.name for skill in one.skills] == ["recursive-lean-proof"] for one in mine
    )
    assert len(theirs) == 6
    assert not any(one.skills for one in theirs)
    # Every turn's session closed with its `turn` call, rather than piling up to the end;
    # the problem was fetched in the one session of its own the run holds itself.
    assert len(sessions) == 18
    assert max(worker.opened) <= 4  # type: ignore[attr-defined]
    assert names.count(("turn", 2)) == len(mine) - 1

    # The scratch directory the worktrees were in went with the run that made it.
    assert not worktrees.exists()


def test_an_interrupted_run_resumes_its_dag_from_the_journal(
    loaded: Any, tmp_path: Path, offline: list[Any]
) -> None:
    project = _problem(tmp_path)
    journal = tmp_path / "epic" / "journal.jsonl"
    params = {"max_depth": "1", "max_nodes": "3"}
    loaded.calls.clear()
    worker, reviewer, _ = _agents(loaded, interrupt="root")

    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(
            run_fake(
                loaded.entry,
                "Prove that the main theorem holds.",
                agents={"worker": worker, "reviewer": reviewer},
                params=params,
                local=DirEnv(project, tmp_path / "scratch"),
                journal=journal,
            )
        )

    # The children were accepted; any integration the stop cut short waits for the resume.
    first = _run_dir(project)
    nodes = _dag(project)
    assert nodes["root.alpha-a1"]["status"] in {"integrating", "proved"}
    assert nodes["root.beta-a1"]["status"] in {"integrating", "proved"}
    assert nodes["root"]["status"] == "lean-review"
    root_worktree = Path(nodes["root"]["worktree"])
    assert (root_worktree / "Root.lean").is_file()
    assert not (project / "Root.lean").exists()
    planned = [one for one in loaded.calls if one["flow"] == "gen-plan"]
    assert len(planned) == 3

    worker, reviewer, turns = _agents(loaded)
    asyncio.run(
        run_fake(
            loaded.entry,
            "Prove that the main theorem holds.",
            agents={"worker": worker, "reviewer": reviewer},
            params=params,
            local=DirEnv(project, tmp_path / "scratch"),
            journal=journal,
            resume=True,
        )
    )

    # The same run picked up its DAG: nothing replanned, only the root formalized again,
    # in the worktree it had.
    assert _run_dir(project) == first
    assert {one["status"] for one in _dag(project).values()} == {"proved"}
    assert [one["flow"] for one in loaded.calls].count("gen-plan") == 3
    [rebuilt] = [one for one in loaded.calls if one["flow"] == "rlcr"][3:]
    assert rebuilt["workspace"] == str(root_worktree)
    assert not any(schema is loaded.models.NaturalProof for _, schema, _, _ in turns)
    # The problem frozen by the first run is the one the resumed run works on.
    assert not any(schema is loaded.models.FetchedProblem for _, schema, _, _ in turns)
    assert len(offline) == 2
    assert (project / "Root.lean").is_file()
    assert git(project, "status", "--porcelain") == ""


def test_an_interface_mismatch_is_raised_rather_than_planned_around(
    loaded: Any, tmp_path: Path
) -> None:
    from hmz.flows import HarnessThrottled, PermissionTooNarrow

    class Anything(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")

    class Failing:
        expected_params = Anything

        def __init__(self, error: Exception) -> None:
            self.error = error

        async def __call__(self, *_: Any, **__: Any) -> None:
            raise self.error

    project = tmp_path / "mismatch_problem"
    project.mkdir()
    runtime = runtime_for(loaded, project, tmp_path, "mismatch fixture")
    node = runtime.store.ensure(
        "root", parent=None, depth=0, title="Root", statement="Root theorem"
    )

    with patch.object(
        loaded.runtime, "load", return_value=Failing(HarnessThrottled("quota"))
    ):
        draft = asyncio.run(runtime._accepted_plan(node, "None."))
    assert draft.name == "plan-draft-v1.md"
    assert "direct plan unavailable" in node.message

    with (
        patch.object(
            loaded.runtime,
            "load",
            return_value=Failing(PermissionTooNarrow("planner needs more")),
        ),
        pytest.raises(PermissionTooNarrow),
    ):
        asyncio.run(runtime._accepted_plan(node, "None."))


def test_a_turn_is_a_call_of_its_own_whose_session_closes_with_it(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hmz.flows import HarnessRefused, HarnessThrottled

    monkeypatch.setattr(loaded.runtime, "FAILED_PAUSE", 0)
    project = tmp_path / "turns_problem"
    project.mkdir()
    local = DirEnv(project, tmp_path / "scratch")

    def failing(error: Exception) -> FakeAgentDriver:
        def reply(prompt: str, *, output_schema: Any, session: Any) -> Any:
            raise error

        return FakeAgentDriver(HarnessKind.CODEX, reply=reply)

    def turn(driver: FakeAgentDriver, schema: str) -> Any:
        return asyncio.run(
            run_fake(
                loaded.turn,
                "audit it",
                agents={"reviewer": driver},
                params={"role": "reviewer", "schema_name": schema},
                local=local,
            )
        )

    # A turn the harness failed at is answered with nothing, as a suppressed one was.
    throttled = failing(HarnessThrottled("429"))
    assert turn(throttled, "NaturalAudit") is None
    assert turn(throttled, "") == ""
    with pytest.raises(HarnessRefused):
        turn(failing(HarnessRefused("logged out")), "NaturalAudit")

    # One that answered is read as the schema asked for, and its session is over.
    passed = {
        "reference_use": REFERENCE_USE,
        "acceptable": True,
        "first_invalid_step": "",
        "required_changes": [],
    }
    answering = FakeAgentDriver(HarnessKind.CODEX, reply=passed)
    audit = turn(answering, "NaturalAudit")
    assert isinstance(audit, loaded.models.NaturalAudit)
    assert audit.passed
    [session] = answering.sessions
    assert session.closed
    assert session.skills[0].name == "recursive-lean-proof"


def test_humanize1_is_handed_the_skill_rules_in_its_plans(
    loaded: Any, tmp_path: Path
) -> None:
    project = tmp_path / "discipline_problem"
    project.mkdir()
    runtime = runtime_for(loaded, project, tmp_path, "discipline fixture")
    node = loaded.models.NodeRecord(
        id="root", title="Root", statement="True", attempts=1
    )
    plan = runtime._implementation_plan(
        node,
        accepted_plan=project / "plan.md",
        natural_path=project / "proof.md",
        children="- None; this node is atomic.",
    ).read_text()
    assert "## Recursive Lean proof discipline" in plan
    assert "Keep the mathematical statement fixed." in plan
    assert not plan.rstrip().endswith("---")


def test_a_run_needs_a_git_root_and_a_comparator(loaded: Any, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="root of a clean Lean git repository"):
        asyncio.run(
            run_fake(
                loaded.entry,
                "Prove it.",
                local=DirEnv(plain, tmp_path / "scratch"),
            )
        )
    project = repository(tmp_path / "no_comparator")
    with pytest.raises(ValueError, match="comparator script not found"):
        asyncio.run(
            run_fake(
                loaded.entry,
                "Prove it.",
                local=DirEnv(project, tmp_path / "scratch"),
            )
        )
