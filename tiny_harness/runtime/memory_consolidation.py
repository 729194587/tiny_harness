"""Transactional snapshot and replacement storage for Memory consolidation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tiny_harness.runtime.memory_store import (
    MEMORY_ARCHIVE_DIRECTORY,
    MEMORY_DIRECTORY,
    CONSOLIDATION_STAGING_PREFIX,
    MemoryBoundaryError,
    MemoryCandidate,
    MemoryConsolidationError,
    MemoryError,
    MemoryFormatError,
    _atomic_write_text,
    _ensure_memory_root,
    _filename_for_memory_name,
    _manifest_from,
    _memory_document,
    _read_complete_memory,
    _resolve_inside,
    discover_memories,
    rebuild_memory_index,
)


@dataclass(frozen=True)
class ConsolidationMemory:
    """One complete Memory captured in an immutable consolidation snapshot."""

    filename: str
    name: str
    memory_type: str
    description: str
    body: str
    modified_at: str
    modified_ns: int
    content_hash: str


@dataclass(frozen=True)
class MemoryConsolidationSnapshot:
    """All active Memories and their optimistic-concurrency fingerprint."""

    workspace: Path
    memories: tuple[ConsolidationMemory, ...]
    fingerprint: str


@dataclass(frozen=True)
class MemoryConsolidationStage:
    """Validated replacement documents held outside active discovery."""

    path: Path
    filenames: tuple[str, ...]


@dataclass(frozen=True)
class MemoryConsolidationCommit:
    """Metadata-only result of an archive-and-replace transaction."""

    archive_path: Path
    archived: int
    activated: int


@dataclass(frozen=True)
class MemoryConsolidationResult:
    """Outcome of an optional consolidation attempt."""

    outcome: str
    before_count: int
    after_count: int
    archive_created: bool
    reason: str | None = None


def _snapshot_fingerprint(memories: Iterable[ConsolidationMemory]) -> str:
    records = [
        {
            "filename": memory.filename,
            "modified_ns": memory.modified_ns,
            "content_hash": memory.content_hash,
        }
        for memory in sorted(memories, key=lambda item: item.filename.casefold())
    ]
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_memories_for_consolidation(
    workspace: Path,
) -> MemoryConsolidationSnapshot:
    """Capture all active valid Memories for optimistic consolidation."""

    catalog = discover_memories(workspace)
    if catalog.issues:
        raise MemoryConsolidationError(
            "Cannot consolidate while Memory discovery has issues"
        )
    if not catalog.manifests:
        return MemoryConsolidationSnapshot(
            workspace=catalog.workspace,
            memories=(),
            fingerprint=_snapshot_fingerprint(()),
        )
    memory_root = _resolve_inside(
        catalog.workspace / MEMORY_DIRECTORY,
        catalog.workspace,
    )
    memories: list[ConsolidationMemory] = []
    for manifest in catalog.manifests.values():
        try:
            resolved = _resolve_inside(manifest.path, catalog.workspace)
            if not resolved.is_relative_to(memory_root) or not resolved.is_file():
                raise MemoryBoundaryError("Memory path escapes Memory store")
            raw, body = _read_complete_memory(resolved)
        except MemoryError as error:
            raise MemoryConsolidationError(
                f"Cannot snapshot Memory: {manifest.filename}"
            ) from error
        modified_at = (
            datetime.fromtimestamp(
                manifest.modified_ns / 1_000_000_000,
                tz=timezone.utc,
            )
            .isoformat()
            .replace("+00:00", "Z")
        )
        memories.append(
            ConsolidationMemory(
                filename=manifest.filename,
                name=manifest.name,
                memory_type=manifest.memory_type,
                description=manifest.description,
                body=body,
                modified_at=modified_at,
                modified_ns=manifest.modified_ns,
                content_hash=hashlib.sha256(raw).hexdigest(),
            )
        )
    snapshot = tuple(memories)
    return MemoryConsolidationSnapshot(
        workspace=catalog.workspace,
        memories=snapshot,
        fingerprint=_snapshot_fingerprint(snapshot),
    )


def _validate_staging_path(path: Path, memory_root: Path) -> Path:
    resolved = _resolve_inside(path, memory_root)
    if (
        resolved.parent != memory_root
        or not resolved.name.startswith(CONSOLIDATION_STAGING_PREFIX)
        or not resolved.is_dir()
    ):
        raise MemoryBoundaryError("Invalid Memory consolidation staging path")
    return resolved


def rollback_consolidation(
    workspace: Path,
    *,
    stage_path: Path | None = None,
    archive_path: Path | None = None,
    activated_filenames: Iterable[str] = (),
    rebuild_index: Callable[[Path], Path] = rebuild_memory_index,
) -> None:
    """Best-effort removal of staging and restoration of archived active files."""

    workspace = workspace.resolve(strict=True)
    memory_root = _ensure_memory_root(workspace)
    activated = tuple(activated_filenames)
    for filename in activated:
        if Path(filename).name != filename or not filename.endswith(".md"):
            raise MemoryBoundaryError("Invalid activated Memory filename")
        (memory_root / filename).unlink(missing_ok=True)

    if archive_path is not None and archive_path.exists():
        archive = _resolve_inside(archive_path, memory_root)
        expected_parent = memory_root / MEMORY_ARCHIVE_DIRECTORY
        if archive.parent != expected_parent.resolve(strict=True):
            raise MemoryBoundaryError("Invalid Memory consolidation archive path")
        for archived in archive.iterdir():
            if archived.suffix.lower() != ".md" or not archived.is_file():
                continue
            destination = memory_root / archived.name
            if destination.exists():
                raise MemoryConsolidationError(
                    "Cannot restore archived Memory over a concurrent file"
                )
            os.replace(archived, destination)
        try:
            archive.rmdir()
        except OSError:
            pass

    if stage_path is not None and stage_path.exists():
        stage = _validate_staging_path(stage_path, memory_root)
        for staged in stage.iterdir():
            if staged.is_file():
                staged.unlink(missing_ok=True)
        stage.rmdir()

    rebuild_index(workspace)


def stage_consolidated_memories(
    workspace: Path,
    candidates: Iterable[MemoryCandidate],
) -> MemoryConsolidationStage:
    """Write and revalidate a complete replacement set outside discovery."""

    proposed = tuple(candidates)
    if not proposed:
        raise MemoryConsolidationError("Cannot stage an empty replacement set")
    memory_root = _ensure_memory_root(workspace)
    stage_path = memory_root / f"{CONSOLIDATION_STAGING_PREFIX}{uuid4().hex}"
    stage_path.mkdir()
    try:
        stage = _validate_staging_path(stage_path, memory_root)
        filenames: list[str] = []
        for candidate in proposed:
            try:
                filename = _filename_for_memory_name(candidate.name)
            except MemoryFormatError as error:
                raise MemoryConsolidationError(str(error)) from error
            destination = stage / filename
            _atomic_write_text(destination, _memory_document(candidate))
            manifest = _manifest_from(
                destination,
                destination.stat().st_mtime_ns,
            )
            _, body = _read_complete_memory(destination)
            if (
                manifest.name != candidate.name
                or manifest.memory_type != candidate.memory_type
                or manifest.description != candidate.description
                or body != candidate.body
            ):
                raise MemoryConsolidationError(
                    "Staged Memory did not round-trip validation"
                )
            filenames.append(filename)
        return MemoryConsolidationStage(stage, tuple(filenames))
    except Exception:
        rollback_consolidation(workspace, stage_path=stage_path)
        raise


def commit_consolidated_memories(
    snapshot: MemoryConsolidationSnapshot,
    stage: MemoryConsolidationStage,
    *,
    rebuild_index: Callable[[Path], Path] = rebuild_memory_index,
) -> MemoryConsolidationCommit:
    """Archive active files and activate staging after a final overlap check."""

    current = snapshot_memories_for_consolidation(snapshot.workspace)
    if current.fingerprint != snapshot.fingerprint:
        rollback_consolidation(
            snapshot.workspace,
            stage_path=stage.path,
            rebuild_index=rebuild_index,
        )
        raise MemoryConsolidationError("Memory changed before consolidation commit")

    memory_root = _ensure_memory_root(snapshot.workspace)
    staging = _validate_staging_path(stage.path, memory_root)
    archive_root = memory_root / MEMORY_ARCHIVE_DIRECTORY
    if archive_root.exists():
        archive_root = _resolve_inside(archive_root, memory_root)
        if not archive_root.is_dir():
            rollback_consolidation(
                snapshot.workspace,
                stage_path=staging,
                rebuild_index=rebuild_index,
            )
            raise MemoryConsolidationError(
                "Memory archive location must be a directory"
            )
    else:
        archive_root.mkdir()
        archive_root = archive_root.resolve(strict=True)
    archive_name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid4().hex}"
    )
    archive_path = archive_root / archive_name
    archive_path.mkdir()

    activated: list[str] = []
    archived = 0
    try:
        for memory in snapshot.memories:
            source = memory_root / memory.filename
            if not source.exists():
                raise MemoryConsolidationError(
                    "Active Memory disappeared during consolidation"
                )
            os.replace(source, archive_path / memory.filename)
            archived += 1

        for filename in stage.filenames:
            source = staging / filename
            destination = memory_root / filename
            if destination.exists():
                raise MemoryConsolidationError(
                    "Consolidated Memory conflicts with a concurrent file"
                )
            os.replace(source, destination)
            activated.append(filename)

        rebuild_index(snapshot.workspace)
        staging.rmdir()
        return MemoryConsolidationCommit(
            archive_path=archive_path,
            archived=archived,
            activated=len(activated),
        )
    except Exception as error:
        try:
            rollback_consolidation(
                snapshot.workspace,
                stage_path=staging,
                archive_path=archive_path,
                activated_filenames=activated,
                rebuild_index=rebuild_index,
            )
        except Exception as rollback_error:
            raise MemoryConsolidationError(
                "Memory consolidation and rollback both failed"
            ) from rollback_error
        if isinstance(error, MemoryError):
            raise
        raise MemoryConsolidationError("Memory consolidation commit failed") from error
