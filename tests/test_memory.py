import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.runtime.memory import (
    MEMORY_CATALOG_MARKER,
    RELEVANT_MEMORY_MARKER,
    MemoryCandidate,
    MemoryConsolidationError,
    MemoryExtractionError,
    commit_consolidated_memories,
    consolidate_memories_if_needed,
    discover_memories,
    format_memory_catalog,
    keyword_memory_selection,
    list_memories,
    load_relevant_memories,
    parse_extracted_memories,
    parse_consolidated_memories,
    parse_memory_selection,
    select_relevant_memories,
    snapshot_memories_for_consolidation,
    stage_consolidated_memories,
    upsert_memory_markers,
    write_memories,
)


class MemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_memory(
        self,
        filename: str,
        *,
        name: str | None = None,
        description: str = "Useful persistent fact",
        memory_type: str = "project",
        body: str = "PRIVATE_MEMORY_BODY",
        modified_ns: int | None = None,
    ) -> Path:
        path = self.workspace / ".tinyharness" / "memory" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n"
            f"name: {name or path.stem}\n"
            f"description: {description}\n"
            f"type: {memory_type}\n"
            "---\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        if modified_ns is not None:
            os.utime(path, ns=(modified_ns, modified_ns))
        return path

    def test_empty_workspace_has_no_memory_side_effect(self) -> None:
        catalog = discover_memories(self.workspace)

        self.assertEqual(catalog.manifests, {})
        self.assertEqual(catalog.issues, ())
        self.assertFalse((self.workspace / ".tinyharness").exists())

    def test_discovery_is_recent_first_and_metadata_only(self) -> None:
        self.write_memory(
            "older.md",
            name="older",
            description="Older project fact",
            body="OLDER_PRIVATE_BODY",
            modified_ns=1_000_000_000,
        )
        self.write_memory(
            "newer.md",
            name="newer",
            description="Newer user preference",
            memory_type="user",
            body="NEWER_PRIVATE_BODY",
            modified_ns=2_000_000_000,
        )

        catalog = discover_memories(self.workspace)

        self.assertEqual(tuple(catalog.manifests), ("newer.md", "older.md"))
        serialized = json.dumps(list_memories(catalog), ensure_ascii=False)
        self.assertIn("Newer user preference", serialized)
        self.assertNotIn("PRIVATE_BODY", serialized)
        self.assertNotIn("PRIVATE_BODY", format_memory_catalog(catalog))

    def test_invalid_and_duplicate_names_are_fail_closed(self) -> None:
        self.write_memory("a.md", name="shared", body="FIRST")
        self.write_memory("b.md", name="shared", body="SECOND")
        self.write_memory("c.md", name="shared", body="THIRD")
        self.write_memory("bad-type.md", memory_type="plan")
        self.write_memory("valid.md", name="valid")

        catalog = discover_memories(self.workspace)

        self.assertEqual(tuple(catalog.manifests), ("valid.md",))
        duplicate_issues = [
            issue
            for issue in catalog.issues
            if "Duplicate Memory name" in issue.reason
        ]
        self.assertEqual(len(duplicate_issues), 3)
        self.assertTrue(
            any("Memory type must be one of" in issue.reason for issue in catalog.issues)
        )

    def test_scan_limit_uses_mtime_order(self) -> None:
        self.write_memory("old.md", name="old", modified_ns=1_000_000_000)
        self.write_memory("middle.md", name="middle", modified_ns=2_000_000_000)
        self.write_memory("new.md", name="new", modified_ns=3_000_000_000)

        catalog = discover_memories(self.workspace, max_memories=2)

        self.assertEqual(tuple(catalog.manifests), ("new.md", "middle.md"))
        self.assertTrue(
            any("Memory scan limit reached: 2" in issue.reason for issue in catalog.issues)
        )

    def test_llm_selection_uses_exact_filenames_and_keyword_fallback(self) -> None:
        self.write_memory(
            "tabs.md",
            name="tabs",
            description="User prefers tabs for indentation",
            memory_type="user",
        )
        self.write_memory(
            "database.md",
            name="database",
            description="Project database migration notes",
        )
        catalog = discover_memories(self.workspace)
        calls = []

        def complete(messages, tools):
            calls.append((messages, tools))
            return ModelResponse(
                '{"selected_memories":["tabs.md"]}',
                None,
                [],
                "stop",
            )

        selection = select_relevant_memories(
            catalog,
            [{"role": "user", "content": "Use my indentation preference"}],
            "Use my indentation preference",
            complete,
        )

        self.assertEqual(selection.filenames, ("tabs.md",))
        self.assertEqual(selection.method, "llm")
        self.assertEqual(calls[0][1], [])
        self.assertNotIn("PRIVATE_MEMORY_BODY", json.dumps(calls[0][0]))
        self.assertEqual(
            keyword_memory_selection(catalog, "please use tabs"),
            ("tabs.md",),
        )
        with self.assertRaisesRegex(Exception, "unknown filename"):
            parse_memory_selection(
                '{"selected_memories":["../outside.md"]}',
                catalog,
            )

    def test_context_budget_skips_side_query_and_uses_keywords(self) -> None:
        self.write_memory(
            "tabs.md",
            name="tabs",
            description="User prefers tabs",
            memory_type="user",
        )
        catalog = discover_memories(self.workspace)

        selection = select_relevant_memories(
            catalog,
            [{"role": "user", "content": "use tabs"}],
            "use tabs",
            lambda *_: self.fail("side query must not run"),
            max_context_chars=10,
        )

        self.assertEqual(selection.filenames, ("tabs.md",))
        self.assertEqual(selection.method, "keyword")
        self.assertEqual(selection.failure_type, "ContextLimitError")

    def test_relevant_content_is_bounded_and_marked_untrusted(self) -> None:
        self.write_memory(
            "large.md",
            name="large",
            body="BODY_SENTINEL\n" + "X" * 8_000,
        )
        catalog = discover_memories(self.workspace)

        loaded = load_relevant_memories(catalog, ["large.md"])
        messages = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "task"},
        ]
        upsert_memory_markers(messages, catalog, loaded)

        self.assertIn("BODY_SENTINEL", loaded.content)
        self.assertIn("<memory-truncated/>", loaded.content)
        self.assertIn("untrusted historical data", loaded.content)
        catalog_marker = next(
            message
            for message in messages
            if message.get("name") == MEMORY_CATALOG_MARKER
        )
        relevant_marker = next(
            message
            for message in messages
            if message.get("name") == RELEVANT_MEMORY_MARKER
        )
        self.assertEqual(catalog_marker["role"], "system")
        self.assertEqual(relevant_marker["role"], "user")
        self.assertNotIn("BODY_SENTINEL", catalog_marker["content"])

    def test_extractor_contract_and_atomic_write_rebuild_index(self) -> None:
        catalog = discover_memories(self.workspace)
        candidates = parse_extracted_memories(
            json.dumps(
                {
                    "memories": [
                        {
                            "name": "user-tabs",
                            "type": "user",
                            "description": "User prefers tabs",
                            "body": "Use tabs for indentation.",
                        }
                    ]
                }
            )
        )

        result = write_memories(self.workspace, catalog, candidates)

        self.assertEqual(result.written, 1)
        memory_path = (
            self.workspace / ".tinyharness" / "memory" / "user-tabs.md"
        )
        index_path = memory_path.parent / "MEMORY.md"
        self.assertIn("Use tabs", memory_path.read_text(encoding="utf-8"))
        self.assertIn("user-tabs.md", index_path.read_text(encoding="utf-8"))
        second = write_memories(self.workspace, catalog, candidates)
        self.assertEqual((second.written, second.skipped), (0, 1))

        with self.assertRaises(MemoryExtractionError):
            parse_extracted_memories(
                '{"memories":[{"name":"../bad","type":"user",'
                '"description":"bad","body":"bad"}]}'
            )
        with self.assertRaisesRegex(MemoryExtractionError, "derived index"):
            parse_extracted_memories(
                '{"memories":[{"name":"MEMORY","type":"user",'
                '"description":"bad","body":"bad"}]}'
            )

    def test_write_skips_when_memory_changed_after_discovery(self) -> None:
        stale_catalog = discover_memories(self.workspace)
        self.write_memory(
            "main-agent.md",
            name="main-agent",
            body="MAIN_AGENT_WRITE",
        )
        candidate = MemoryCandidate(
            "extractor-write",
            "project",
            "Extractor proposal",
            "EXTRACTOR_WRITE",
        )

        result = write_memories(self.workspace, stale_catalog, [candidate])

        self.assertEqual((result.written, result.skipped), (0, 1))
        self.assertFalse(
            (
                self.workspace
                / ".tinyharness"
                / "memory"
                / "extractor-write.md"
            ).exists()
        )

    def test_load_rechecks_symlink_escape_after_discovery(self) -> None:
        memory = self.write_memory("changing.md", name="changing", body="SAFE")
        outside = self.root / "outside.md"
        outside.write_text("OUTSIDE_SENTINEL", encoding="utf-8")
        catalog = discover_memories(self.workspace)
        memory.unlink()
        try:
            memory.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"symbolic links unavailable: {error}")

        loaded = load_relevant_memories(catalog, ["changing.md"])

        self.assertEqual(loaded.content, "")
        self.assertEqual(len(loaded.issues), 1)
        self.assertIn("escapes workspace", loaded.issues[0].reason)

    def test_consolidation_parser_allows_equal_count_but_not_growth(self) -> None:
        payload = {
            "memories": [
                {
                    "name": "one",
                    "type": "project",
                    "description": "First durable fact",
                    "body": "ONE",
                },
                {
                    "name": "two",
                    "type": "reference",
                    "description": "Second durable fact",
                    "body": "TWO",
                },
            ]
        }

        candidates = parse_consolidated_memories(
            json.dumps(payload),
            before_count=2,
        )

        self.assertEqual(len(candidates), 2)
        with self.assertRaisesRegex(
            MemoryConsolidationError,
            "must not increase",
        ):
            parse_consolidated_memories(
                json.dumps(payload),
                before_count=1,
            )
        with self.assertRaisesRegex(MemoryConsolidationError, "cannot become empty"):
            parse_consolidated_memories(
                '{"memories":[]}',
                before_count=2,
            )
        reserved = {"memories": [dict(payload["memories"][0], name="MEMORY")]}
        with self.assertRaisesRegex(MemoryConsolidationError, "derived index"):
            parse_consolidated_memories(
                json.dumps(reserved),
                before_count=1,
            )
        case_collision = {
            "memories": [
                dict(payload["memories"][0], name="Fact"),
                dict(payload["memories"][1], name="fact"),
            ]
        }
        with self.assertRaisesRegex(MemoryConsolidationError, "filenames"):
            parse_consolidated_memories(
                json.dumps(case_collision),
                before_count=2,
            )

    def test_snapshot_has_freshness_and_transaction_dirs_are_not_discovered(self) -> None:
        self.write_memory(
            "fact.md",
            name="fact",
            body="FULL_BODY",
            modified_ns=1_700_000_000_000_000_000,
        )
        memory_root = self.workspace / ".tinyharness" / "memory"
        archive = memory_root / "archive" / "old"
        archive.mkdir(parents=True)
        (archive / "archived.md").write_text("not active", encoding="utf-8")
        staging = memory_root / ".consolidate-staging-test"
        staging.mkdir()
        (staging / "staged.md").write_text("not active", encoding="utf-8")

        snapshot = snapshot_memories_for_consolidation(self.workspace)
        catalog = discover_memories(self.workspace)

        self.assertEqual(tuple(catalog.manifests), ("fact.md",))
        self.assertEqual(len(snapshot.memories), 1)
        memory = snapshot.memories[0]
        self.assertEqual(memory.body, "FULL_BODY")
        self.assertTrue(memory.modified_at.endswith("Z"))
        self.assertEqual(len(memory.content_hash), 64)
        self.assertEqual(len(snapshot.fingerprint), 64)

    def test_consolidation_archives_and_replaces_active_set(self) -> None:
        self.write_memory("one.md", name="one", body="ONE")
        self.write_memory("two.md", name="two", body="TWO")
        calls = []

        class Logger:
            def __init__(self) -> None:
                self.events = []

            def emit(self, event_type, data=None) -> None:
                self.events.append((event_type.value, dict(data or {})))

        logger = Logger()

        def complete(messages, tools):
            calls.append((messages, tools))
            return ModelResponse(
                json.dumps(
                    {
                        "memories": [
                            {
                                "name": "combined",
                                "type": "project",
                                "description": "Combined durable facts",
                                "body": "ONE\n\nTWO",
                            }
                        ]
                    }
                ),
                None,
                [],
                "stop",
            )

        result = consolidate_memories_if_needed(
            self.workspace,
            complete,
            logger,
            threshold=2,
        )

        self.assertEqual(result.outcome, "completed")
        self.assertEqual((result.before_count, result.after_count), (2, 1))
        self.assertEqual(calls[0][1], [])
        request_text = json.dumps(calls[0][0], ensure_ascii=False)
        self.assertIn("modified_at", request_text)
        self.assertNotIn("modified_ns", request_text)
        self.assertNotIn("content_hash", request_text)
        catalog = discover_memories(self.workspace)
        self.assertEqual(tuple(catalog.manifests), ("combined.md",))
        archive_root = self.workspace / ".tinyharness" / "memory" / "archive"
        archive_dirs = list(archive_root.iterdir())
        self.assertEqual(len(archive_dirs), 1)
        self.assertEqual(
            {item.name for item in archive_dirs[0].iterdir()},
            {"one.md", "two.md"},
        )
        self.assertIn(
            "memory_consolidation_completed",
            [event_type for event_type, _ in logger.events],
        )

    def test_commit_overlap_guard_removes_staging_without_replacing(self) -> None:
        original = self.write_memory("one.md", name="one", body="ORIGINAL")
        snapshot = snapshot_memories_for_consolidation(self.workspace)
        stage = stage_consolidated_memories(
            self.workspace,
            [MemoryCandidate("replacement", "project", "Replacement", "NEW")],
        )
        original.write_text(
            original.read_text(encoding="utf-8") + "CONCURRENT\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            MemoryConsolidationError,
            "changed before consolidation commit",
        ):
            commit_consolidated_memories(snapshot, stage)

        self.assertFalse(stage.path.exists())
        self.assertTrue(original.exists())
        self.assertFalse(
            (original.parent / "replacement.md").exists()
        )

    def test_model_time_overlap_fails_before_staging(self) -> None:
        original = self.write_memory("one.md", name="one", body="ORIGINAL")

        class Logger:
            def __init__(self) -> None:
                self.events = []

            def emit(self, event_type, data=None) -> None:
                self.events.append((event_type.value, dict(data or {})))

        logger = Logger()

        def complete(messages, tools):
            del messages, tools
            original.write_text(
                original.read_text(encoding="utf-8") + "CONCURRENT\n",
                encoding="utf-8",
            )
            return ModelResponse(
                json.dumps(
                    {
                        "memories": [
                            {
                                "name": "replacement",
                                "type": "project",
                                "description": "Replacement",
                                "body": "NEW",
                            }
                        ]
                    }
                ),
                None,
                [],
                "stop",
            )

        result = consolidate_memories_if_needed(
            self.workspace,
            complete,
            logger,
            threshold=1,
        )

        self.assertEqual(result.outcome, "failed")
        self.assertTrue(original.exists())
        self.assertFalse((original.parent / "replacement.md").exists())
        self.assertFalse(
            any(
                path.name.startswith(".consolidate-staging-")
                for path in original.parent.iterdir()
            )
        )
        self.assertIn(
            "memory_consolidation_failed",
            [event_type for event_type, _ in logger.events],
        )

    def test_commit_failure_rolls_back_archived_and_activated_files(self) -> None:
        original = self.write_memory("one.md", name="one", body="ORIGINAL")
        snapshot = snapshot_memories_for_consolidation(self.workspace)
        stage = stage_consolidated_memories(
            self.workspace,
            [MemoryCandidate("replacement", "project", "Replacement", "NEW")],
        )
        from tiny_harness.runtime import memory as memory_module

        real_rebuild = memory_module.rebuild_memory_index
        calls = 0

        def fail_first_rebuild(workspace):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated index failure")
            return real_rebuild(workspace)

        with patch.object(
            memory_module,
            "rebuild_memory_index",
            side_effect=fail_first_rebuild,
        ):
            with self.assertRaisesRegex(
                MemoryConsolidationError,
                "commit failed",
            ):
                commit_consolidated_memories(snapshot, stage)

        self.assertTrue(original.exists())
        self.assertIn("ORIGINAL", original.read_text(encoding="utf-8"))
        self.assertFalse((original.parent / "replacement.md").exists())
        self.assertFalse(stage.path.exists())


if __name__ == "__main__":
    unittest.main()
