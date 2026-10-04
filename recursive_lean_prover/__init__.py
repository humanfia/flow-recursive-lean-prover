"""Recursively plan, prove, compare, review, and catalogue Lean theorems."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Literal, NotRequired

from _recursive_lean.runtime import Runtime, take_turn
from hmz.flows import (
    Agent,
    AgentCollection,
    BashEnvMixin,
    EnvCollection,
    FilesEnvMixin,
    FlowContext,
    FlowParams,
    GitWorktreeEnvMixin,
    LocalEnv,
    Permission,
    PermissionKind,
    PermissionRequestHookAgentMixin,
    RequirementError,
    ScratchDirEnvMixin,
    flow,
    load,
)
from pydantic import ConfigDict, Field, field_validator, model_validator

MIN_RECURSIVE_NODES = 3

RLCR = "humanize1:rlcr"

# The flow's own sessions rerun the comparator, whose Lean builds write through the linked
# `.lake/packages`, and commit repairs in linked worktrees whose Git metadata is the primary
# checkout's; both roles may read the internet. Holding this much also covers what the
# humanize1 roles they are handed to ask, which run under their own declarations.
ROLE_PERMISSION = Permission(
    local=PermissionKind.ALL,
    user=PermissionKind.ALL,
    system=PermissionKind.READ,
    online=PermissionKind.ALL,
)


class Worker(Agent, PermissionRequestHookAgentMixin):
    """Plans, proves, formalizes and repairs; RLCR's builder, whose guards hook permission."""

    _permission = ROLE_PERMISSION
    _skills = ("recursive-lean-proof",)


class Reviewer(Agent):
    """Audits every proof, decomposition and Lean candidate, and reruns the comparator."""

    _permission = ROLE_PERMISSION
    _skills = ("recursive-lean-proof",)


class Workspace(
    LocalEnv,
    BashEnvMixin,
    FilesEnvMixin,
    GitWorktreeEnvMixin,
    ScratchDirEnvMixin,
):
    """The Lean repository: commands, files, node worktrees, and a scratch directory for them."""


class Agents(AgentCollection):
    worker: Worker
    reviewer: Reviewer


class Envs(EnvCollection):
    workspace: Workspace


class Turning(AgentCollection):
    worker: NotRequired[Worker]
    reviewer: NotRequired[Reviewer]


class Config(FlowParams):
    model_config = ConfigDict(frozen=True)

    max_depth: int = Field(
        default=2,
        ge=0,
        le=6,
        description="deepest recursive subproblem level; the root is level zero",
    )
    max_children: int = Field(
        default=4,
        ge=2,
        le=12,
        description="most direct subproblems one theorem may activate",
    )
    max_parallel_children: int = Field(
        default=24,
        ge=1,
        le=200,
        description=(
            "worker-pool size for the dependency-ready nodes of one DAG frontier"
        ),
    )
    max_nodes: int = Field(
        default=24,
        ge=1,
        le=200,
        description="hard bound on all theorem and subproblem nodes in one run",
    )
    node_attempts: int = Field(
        default=2,
        ge=1,
        le=8,
        description=(
            "legacy compatibility setting; after one scaffold exists, correctness "
            "feedback continuously iterates the NL proof"
        ),
    )
    plan_attempts: int = Field(
        default=1,
        ge=1,
        le=1,
        description="exactly one immutable scaffold-plan generation per node",
    )
    natural_proof_attempts: int = Field(
        default=3,
        ge=1,
        le=8,
        description=(
            "natural-language proof/review revisions per batch; all batches continue "
            "from the latest rejected draft until review passes"
        ),
    )
    decomposition_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="attempts to obtain a valid acyclic subproblem decomposition",
    )
    rlcr_rounds: int = Field(
        default=20,
        ge=1,
        le=200,
        description="maximum official humanize1:rlcr rounds per Lean node",
    )
    plan_turn_timeout: float = Field(
        default=3600,
        ge=0,
        description="seconds for one humanize1 planning turn; zero disables it",
    )
    plan_total_timeout: float = Field(
        default=14400,
        ge=0,
        description="seconds for one complete planning phase; zero disables it",
    )
    comparator_timeout: float = Field(
        default=21600,
        ge=1,
        description="seconds allowed for each independent comparator run",
    )
    problem_id: str = Field(
        default="",
        description=(
            "one Lean-Eval problem id; blank derives it from the task URL, workspace "
            "README, or workspace directory"
        ),
    )
    problem_fetch_attempts: int = Field(
        default=3,
        ge=1,
        le=8,
        description="attempts in one dedicated session to fetch one valid problem page",
    )
    artifact_dir: str = Field(
        default=".humanize/recursive-lean-prover",
        description="untracked directory for plans, proofs, DAGs, logs, and run state",
    )
    wiki_dir: str = Field(
        default=".humanize/math-wiki",
        description="Markdown wiki receiving every comparator-approved theorem",
    )
    reference_dir: str = Field(
        default=".humanize/math-reference-library",
        description="untracked cache for the three mandatory reference repositories",
    )
    huggingface_token_env: str = Field(
        default="HF_TOKEN",
        description=(
            "environment variable holding the Hugging Face read token; the value is "
            "never written to params, prompts, manifests, or subprocess arguments"
        ),
    )
    lean_target: str = Field(
        default="",
        description="Lean file the worker must edit; blank lets it infer the project target",
    )
    agent_hidden_files: tuple[str, ...] = Field(
        default=(),
        description=(
            "tracked comparator-only files removed from the main checkout and every "
            "agent worktree before any agent session starts"
        ),
    )
    comparator_command: str = Field(
        default="bash tools/check-with-comparator.sh",
        min_length=1,
        description="argv-style comparator command; node placeholders are supported",
    )
    comparator_success: str = Field(
        default="Your solution is okay!",
        min_length=1,
        description="text that must occur in successful comparator output",
    )
    stop_on_child_failure: bool = Field(
        default=True,
        description="block a parent when any required subproblem exhausts its attempts",
    )

    @field_validator("artifact_dir", "wiki_dir", "reference_dir")
    @classmethod
    def _local_state(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(".humanize/"):
            raise ValueError("must be a relative path below .humanize/")
        if ".." in normalized.split("/"):
            raise ValueError("must not contain '..'")
        return normalized

    @field_validator("problem_id")
    @classmethod
    def _problem_id(cls, value: str) -> str:
        normalized = value.strip()
        if normalized and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", normalized
        ):
            raise ValueError("must be blank or one Lean-Eval problem id")
        return normalized

    @field_validator("huggingface_token_env")
    @classmethod
    def _token_environment_name(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
            raise ValueError("must be an environment-variable name")
        if normalized.casefold() in {
            "codex_home",
            "home",
            "path",
            "pwd",
            "shell",
            "user",
        }:
            raise ValueError("must not repurpose a common system environment variable")
        return normalized

    @field_validator("lean_target")
    @classmethod
    def _relative_lean_target(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("must be blank or a relative path inside the repository")
        if normalized and not normalized.endswith(".lean"):
            raise ValueError("must name a .lean file")
        return normalized

    @field_validator("agent_hidden_files")
    @classmethod
    def _relative_hidden_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            one = value.strip().rstrip("/")
            parts = Path(one).parts
            if (
                not one
                or not parts
                or Path(one).is_absolute()
                or ".." in parts
                or parts[0] == ".git"
            ):
                raise ValueError(
                    "entries must be relative file paths inside the repository and "
                    "outside .git"
                )
            if one in normalized:
                raise ValueError("entries must be unique")
            normalized.append(one)
        return tuple(normalized)

    @model_validator(mode="after")
    def _tree_fits(self) -> Config:
        if self.max_depth and self.max_nodes < MIN_RECURSIVE_NODES:
            raise ValueError("recursive runs need max_nodes >= 3")
        return self


class WorktreeRlcrConfig(FlowParams):
    model_config = ConfigDict(frozen=True)

    plan_file: str = Field(description="absolute immutable implementation plan path")
    max: int = Field(
        default=20,
        ge=1,
        le=200,
        description="maximum official RLCR implementation/review rounds",
    )
    base_branch: str = Field(
        default="",
        description=(
            "exact post-overlay commit retained in the node audit configuration"
        ),
    )
    track_plan_file: bool = False
    push_every_round: bool = False
    skip_impl: bool = False
    skip_quiz: bool = True
    privacy: bool = True
    agent_teams: bool = False
    claude_answer_codex: bool = True


class Turn(FlowParams):
    role: Literal["worker", "reviewer"] = Field(
        description="which agent takes the turn"
    )
    schema_name: str = Field(
        default="",
        description="the model of `_recursive_lean.models` it answers as; blank for text",
    )


def _nested_rlcr_config(config: WorktreeRlcrConfig) -> dict[str, Any]:
    forwarded = config.model_dump()
    forwarded["base_branch"] = ""
    forwarded["skip_code_review"] = True
    return forwarded


def _require_explicit_rlcr_review_skip(rlcr: Any) -> None:
    if "skip_code_review" not in rlcr.expected_params.model_fields:
        raise RequirementError(
            f"{RLCR} is too old: install a humanize1 whose rlcr exposes "
            "skip_code_review, which keeps each node's review isolated"
        )


@flow(
    agents=Agents,
    envs=Envs,
    params=Config,
    resumable=True,
    description=(
        "Fetch one Lean-Eval problem, then recursively prove it with three reference "
        "corpora, RLCR, comparator gates, a live DAG, and a wiki"
    ),
)
async def recursive_lean_prover(
    task: str,
    *,
    agents: Agents,
    envs: Envs,
    params: Config,
    ctx: FlowContext,
) -> None:
    await Runtime(agents, envs, task, params, ctx.state, ctx).execute()


@flow(
    agents=Agents,
    envs=Envs,
    params=WorktreeRlcrConfig,
    name="worktree-rlcr",
    hidden=True,
    resumable=True,
    description="Run official RLCR in one node worktree",
)
async def worktree_rlcr(
    task: str,
    *,
    agents: Agents,
    envs: Envs,
    params: WorktreeRlcrConfig,
    ctx: FlowContext,
) -> str:
    workspace = envs["workspace"]
    worktree = Path(str(workspace.workdir))
    manifest = worktree / "lake-manifest.json"
    ignored, _, _ = await workspace.exec(
        ["git", "check-ignore", "--quiet", "lake-manifest.json"]
    )
    common, found, _ = await workspace.exec(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]
    )
    if not manifest.exists() and ignored == 0 and common == 0 and found.strip():
        source = Path(found.strip()).parent / "lake-manifest.json"
        if source.is_file() and source.resolve() != manifest.resolve():
            shutil.copy2(source, manifest)
    rlcr = load(RLCR)
    _require_explicit_rlcr_review_skip(rlcr)
    return await rlcr(
        task,
        agents={"builder": agents["worker"], "reviewer": agents["reviewer"]},
        envs={"workspace": workspace},
        params=rlcr.expected_params.model_validate(_nested_rlcr_config(params)),
    )


@flow(
    agents=Turning,
    envs=Envs,
    params=Turn,
    name="turn",
    hidden=True,
    description="One turn of one agent in a session of its own, closed as it returns",
)
async def turn(
    task: str,
    *,
    agents: Turning,
    envs: Envs,
    params: Turn,
    ctx: FlowContext,
) -> Any:
    agent = agents.get(params.role)
    if agent is None:
        raise ValueError(
            f"a {params.role} turn was asked for, and no {params.role} given"
        )
    return await take_turn(agent, task, params.schema_name, envs["workspace"])


__all__ = [
    "Agents",
    "Config",
    "Envs",
    "Reviewer",
    "Turn",
    "Turning",
    "Worker",
    "Workspace",
    "WorktreeRlcrConfig",
    "recursive_lean_prover",
    "turn",
    "worktree_rlcr",
]
