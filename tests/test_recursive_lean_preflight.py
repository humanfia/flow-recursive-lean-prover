"""recursive_lean_prover's preflight: three pinned reference snapshots, then one problem.

Nothing here reaches the network: `git clone` is a scripted `subprocess.run`, the problem's
v2 JSON is handed over as the controller would have downloaded it, and the agent that fetches
the problem page is a fake answering in one session.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pydantic
import pytest

from tests.test_recursive_lean_prover import (
    REFERENCE_USE,
    DirEnv,
    fetched_problem,
    flowverse,
    problem_markdown,
    problem_site_data,
    reference_bundle,
    runtime_for,
)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    with flowverse(tmp_path_factory.mktemp("verse")) as made:
        yield made


def _project(tmp_path: Path, name: str = "mihailescu") -> Path:
    project = tmp_path / name
    project.mkdir()
    return project


class FetchAgent:
    """A worker the problem is fetched by: each session it opens, and each turn in one."""

    role = "worker"

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.sessions: list[Any] = []
        self.turns: list[tuple[str, Any, Any]] = []

    async def spawn(self, *, env: Any) -> Any:
        self.sessions.append(SimpleNamespace(env=env))
        return self.sessions[-1]

    async def run(self, prompt: str, *, session: Any, output_schema: Any) -> Any:
        self.turns.append((prompt, session, output_schema))
        return self.answer.model_copy(deep=True)


# ------------------------------------------------------------------------------- schemas


def test_fetched_problem_schema_rejects_a_collection_or_second_problem(
    loaded: Any,
) -> None:
    baseline = fetched_problem(loaded, "mihailescu").model_dump()
    model = loaded.models.FetchedProblem
    with pytest.raises(pydantic.ValidationError, match="canonical leaf URL"):
        model.model_validate(
            baseline | {"source_url": "https://lean-lang.org/eval/problems/"}
        )
    with pytest.raises(pydantic.ValidationError, match="exactly one top-level heading"):
        model.model_validate(
            baseline
            | {"markdown": problem_markdown("mihailescu") + "\n# A second problem\n"}
        )
    with pytest.raises(
        pydantic.ValidationError, match="Problem id rows that all match"
    ):
        model.model_validate(
            baseline
            | {
                "markdown": problem_markdown("mihailescu")
                + "\n| Problem id | `dimitrov` |\n"
            }
        )
    repeated = problem_markdown("mihailescu") + "\n| Problem id | `mihailescu` |\n"
    assert model.model_validate(baseline | {"markdown": repeated}).markdown == repeated


def test_reference_aware_outputs_require_all_three_sources(loaded: Any) -> None:
    fixture = {
        "proof": "A sufficiently detailed numbered proof for the fixture.",
        "key_steps": ["Conclude the fixture."],
        "unresolved": [],
    }
    loaded.models.NaturalProof(reference_use=REFERENCE_USE, **fixture)
    with pytest.raises(pydantic.ValidationError, match="at least 3 items"):
        loaded.models.NaturalProof(reference_use=REFERENCE_USE[:2], **fixture)
    twice = [REFERENCE_USE[0], REFERENCE_USE[0], REFERENCE_USE[1]]
    with pytest.raises(pydantic.ValidationError, match="exactly TauCeti"):
        loaded.models.NaturalProof(reference_use=twice, **fixture)


def test_every_reasoning_prompt_receives_problem_and_reference_context(
    loaded: Any,
) -> None:
    prompts = loaded.prompts
    for prompt in (
        prompts.PLAN_DRAFT,
        prompts.NATURAL_PROOF,
        prompts.NATURAL_AUDIT,
        prompts.DECOMPOSE,
        prompts.DECOMPOSITION_AUDIT,
        prompts.RLCR_LEAN_TASK,
        prompts.LEAN_AUDIT,
        prompts.INTEGRATION_REPAIR,
        prompts.INTEGRATION_AUDIT,
    ):
        assert "{problem_context}" in prompt
        assert "{reference_context}" in prompt
    for field in ("{node_id}", "{node_title}", "{lean_name}", "{lean_statement}"):
        assert field in prompts.NATURAL_AUDIT
    assert "requires_parent_revision" in prompts.NATURAL_AUDIT


def test_child_formalization_removes_unproved_inherited_placeholders(
    loaded: Any,
) -> None:
    task = loaded.prompts.RLCR_LEAN_TASK
    assert "remove that placeholder declaration before comparison" in task
    assert "Never replace it with a fake" in task
    assert "Do not recover or inspect them through Git objects/history" in task
    assert "comparator-only source" in loaded.prompts.FETCH_ONE_PROBLEM


# ------------------------------------------------------------------------- the problem id


def test_problem_id_is_resolved_before_the_agent_session(
    loaded: Any, tmp_path: Path
) -> None:
    infer = loaded.preflight.infer_problem_id
    project = _project(tmp_path, "unexpected-worktree-name")
    (project / "README.md").write_text("# Fixture\n\n- Problem ID: `mihailescu`\n")
    assert infer(project, "", "prove it") == "mihailescu"
    assert infer(project, "mihailescu", "prove it") == "mihailescu"
    with pytest.raises(RuntimeError, match="conflicting Lean-Eval problem ids"):
        infer(
            project,
            "dimitrov",
            "use https://lean-lang.org/eval/problems/mihailescu/",
        )
    bare = _project(tmp_path, "catalan")
    assert infer(bare, "", "prove it") == "catalan"


# -------------------------------------------------------------------- the reference library


def test_reference_download_is_complete_pinned_and_token_safe(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = loaded.preflight
    root = tmp_path / "references"
    helper = tmp_path / "askpass.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o700)
    clones: list[tuple[list[str], dict[str, str]]] = []
    checkouts: dict[Path, Any] = {}
    statuses = 0

    def run(arguments: list[str], **kwargs: Any) -> SimpleNamespace:
        nonlocal statuses
        if "clone" in arguments:
            destination = Path(arguments[-1])
            (destination / ".git").mkdir(parents=True)
            source = next(
                one for one in preflight.REFERENCE_SOURCES if one.url in arguments
            )
            checkouts[destination] = source
            for sentinel in source.sentinels:
                path = destination / sentinel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            clones.append((arguments, kwargs["env"]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if arguments[1:] == ["remote", "get-url", "origin"]:
            checkout = Path(kwargs["cwd"])
            source = checkouts.get(checkout) or next(
                one
                for one in preflight.REFERENCE_SOURCES
                if one.directory == checkout.name
            )
            return SimpleNamespace(returncode=0, stdout=source.url + "\n", stderr="")
        if arguments[1:] == ["status", "--porcelain"]:
            statuses += 1
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")

    monkeypatch.setenv("TEST_HF_TOKEN", "top-secret-token")
    monkeypatch.setattr(preflight.subprocess, "run", run)
    library = preflight.ReferenceLibrary(
        root, huggingface_token_env="TEST_HF_TOKEN", askpass_script=helper
    )
    try:
        bundle = library.prepare()

        assert set(bundle.paths) == {one.name for one in preflight.REFERENCE_SOURCES}
        assert bundle.manifest.is_file()
        first_manifest = bundle.manifest.read_text()
        assert "top-secret-token" not in first_manifest
        assert len(clones) == 3
        assert statuses == 6
        assert all("top-secret-token" not in " ".join(call) for call, _ in clones)
        assert all("TEST_HF_TOKEN" not in environment for _, environment in clones)
        assert all(
            "HUMANIZE_HF_TOKEN" not in environment for _, environment in clones[:2]
        )
        assert clones[-1][1]["HUMANIZE_HF_TOKEN"] == "top-secret-token"
        assert clones[-1][1]["GIT_ASKPASS"] == str(helper.resolve())
        assert all(
            not path.stat().st_mode & 0o222
            for checkout in bundle.paths.values()
            for path in [checkout, *checkout.rglob("*")]
            if not path.is_symlink()
        )

        # A resumed run reuses the pinned snapshots without rescanning them.
        resumed = preflight.ReferenceLibrary(
            root, huggingface_token_env="TEST_HF_TOKEN", askpass_script=helper
        ).prepare()
        assert resumed.commits == bundle.commits
        assert statuses == 6
        assert bundle.manifest.read_text() == first_manifest
        context = bundle.prompt_context()
        for source in ("TauCeti", "lean-pool", "mathlib-internal"):
            assert source in context
    finally:
        for path in [root, *root.rglob("*")]:
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o700)


def test_the_private_snapshot_needs_its_token(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = loaded.preflight
    private = next(one for one in preflight.REFERENCE_SOURCES if one.private)
    monkeypatch.delenv("TEST_HF_TOKEN", raising=False)
    library = preflight.ReferenceLibrary(
        tmp_path / "references", huggingface_token_env="TEST_HF_TOKEN"
    )
    with pytest.raises(RuntimeError, match="TEST_HF_TOKEN is required"):
        library._clone(private, tmp_path / "references" / private.directory)
    assert preflight.ASKPASS.is_file()
    assert os.access(preflight.ASKPASS, os.X_OK)


# --------------------------------------------------------------------------- the problem


def test_dedicated_session_writes_and_reuses_one_problem_markdown(
    loaded: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    runtime = runtime_for(loaded, project, tmp_path, "prove the configured theorem")
    worker = FetchAgent(fetched_problem(loaded, "mihailescu"))
    runtime.worker = worker
    runtime.problem_id = "mihailescu"

    async def scenario() -> tuple[Any, Any]:
        with patch.object(
            runtime,
            "_problem_site_data",
            AsyncMock(return_value=problem_site_data("mihailescu")),
        ):
            first = await runtime._fetched_problem()
            (runtime.run_root / "problem.json").unlink()
            return first, await runtime._fetched_problem()

    first, second = asyncio.run(scenario())

    assert first == second
    assert (runtime.run_root / "problem.json").is_file()
    assert len(worker.sessions) == 1
    assert worker.sessions[0].env is runtime.workspace
    [(prompt, session, schema)] = worker.turns
    assert session is worker.sessions[0]
    assert schema is loaded.models.FetchedProblem
    assert "only permitted problem id: `mihailescu`" in prompt
    written = runtime.problem_path.read_text()
    assert len(re.findall(r"(?m)^# ", written)) == 1
    assert written == first.markdown
    assert "problem-site-data.json" in written


def test_an_interrupted_acquisition_is_not_started_again(
    loaded: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    runtime = runtime_for(loaded, project, tmp_path, "prove it")
    worker = FetchAgent(fetched_problem(loaded, "mihailescu"))
    runtime.worker = worker
    runtime.problem_id = "mihailescu"
    (runtime.run_root / "problem-session.json").write_text('{"status": "started"}\n')

    with (
        patch.object(
            runtime,
            "_problem_site_data",
            AsyncMock(return_value=problem_site_data("mihailescu")),
        ),
        pytest.raises(RuntimeError, match="refusing to start a second session"),
    ):
        asyncio.run(runtime._fetched_problem())
    assert not worker.sessions


def test_problem_candidate_is_checked_against_controller_site_data(
    loaded: Any, tmp_path: Path
) -> None:
    runtime = runtime_for(loaded, _project(tmp_path), tmp_path, "prove it")
    runtime.problem_id = "mihailescu"
    data = problem_site_data("mihailescu")
    fetched = fetched_problem(loaded, "mihailescu")
    assert runtime._problem_authority_feedback(fetched, data) == ""
    data["problem"]["title"] = "A different authoritative title"
    assert "title must be" in runtime._problem_authority_feedback(fetched, data)
    with pytest.raises(RuntimeError, match="wrong problem id"):
        runtime._validate_problem_site_data(problem_site_data("dimitrov"))


def test_bootstrap_prepares_references_then_fetches_and_drops_the_token(
    loaded: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    runtime = runtime_for(
        loaded, project, tmp_path, "prove it", problem_id="mihailescu"
    )
    bundle = reference_bundle(loaded, project / ".humanize" / "math-reference-library")
    monkeypatch.setenv("HF_TOKEN", "bootstrap-only-secret")
    monkeypatch.setattr(loaded.preflight.ReferenceLibrary, "prepare", lambda _: bundle)
    worker = FetchAgent(fetched_problem(loaded, "mihailescu"))
    runtime.worker = worker

    with patch.object(
        runtime,
        "_problem_site_data",
        AsyncMock(return_value=problem_site_data("mihailescu")),
    ):
        fetched = asyncio.run(runtime._bootstrap())

    assert fetched.problem_id == "mihailescu"
    assert runtime.reference_bundle is bundle
    assert "HF_TOKEN" not in os.environ
    assert '"status": "ready"' in (runtime.run_root / "preflight.json").read_text()
    assert str(bundle.manifest) in runtime._reference_context()
    assert str(runtime.problem_path.resolve()) in runtime._problem_context()


def test_bootstrap_finishes_before_the_root_solver_starts(
    loaded: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    runtime = runtime_for(loaded, project, tmp_path, "prove it")
    events: list[str] = []

    async def bootstrap() -> Any:
        events.append("bootstrap")
        return fetched_problem(loaded, "mihailescu")

    async def solve(_: Any) -> Any:
        events.append("solve")
        return loaded.models.SolveResult(ok=True, node_id="root")

    with (
        patch.object(runtime, "_require_git", AsyncMock()),
        patch.object(runtime, "_require_comparator", Mock()),
        patch.object(runtime, "_bootstrap", side_effect=bootstrap),
        patch.object(runtime, "_solve", side_effect=solve),
    ):
        asyncio.run(runtime.execute())

    assert events == ["bootstrap", "solve"]


# --------------------------------------------------------------------------- the run


def test_digest_identity_resumes_when_latest_points_to_another_task(
    loaded: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    first = runtime_for(loaded, project, tmp_path, "prove it")
    other = runtime_for(loaded, project, tmp_path, "prove a different theorem")
    latest = project / ".humanize" / "recursive-lean-prover" / "LATEST"
    latest.write_text(str(other.run_root.relative_to(project)) + "\n")

    resumed = runtime_for(loaded, project, tmp_path, "prove it")

    assert resumed.run_root == first.run_root
    assert other.run_root != first.run_root


def test_untrusted_hmz_run_dir_cannot_escape_artifact_root(
    loaded: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    task = "prove it"
    digest = hashlib.sha256(f"mihailescu\0{task}".encode()).hexdigest()
    runtime = loaded.runtime.Runtime(
        {},
        {"workspace": DirEnv(project, tmp_path / "scratch")},
        task,
        loaded.flow.Config(problem_id="mihailescu"),
        {"version": 1, "task_digest": digest, "run_dir": "."},
    )
    artifact_root = project / ".humanize" / "recursive-lean-prover"
    assert runtime.run_root.is_relative_to(artifact_root)
    assert runtime.run_root != project


def test_reference_evidence_must_point_inside_each_snapshot(
    loaded: Any, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    runtime = runtime_for(loaded, project, tmp_path, "prove it")
    runtime.reference_bundle = reference_bundle(
        loaded, project / ".humanize" / "math-reference-library"
    )
    paths = runtime.reference_bundle.paths
    valid = loaded.models.NaturalProof(
        reference_use=[
            {
                "source": name,
                "queries": ["fixture"],
                "files": [str(path / "README.md")],
                "conclusion": "fixture lookup",
            }
            for name, path in paths.items()
        ],
        proof="A sufficiently detailed numbered proof for the fixture.",
        key_steps=["Conclude the fixture."],
        unresolved=[],
    )
    assert runtime._reference_use_problem(valid) == ""
    invalid = valid.model_dump()
    invalid["reference_use"][0]["files"] = ["/tmp/not-in-snapshot"]
    assert "did not cite" in runtime._reference_use_problem(
        loaded.models.NaturalProof.model_validate(invalid)
    )
