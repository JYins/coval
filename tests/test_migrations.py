"""Tests for versioned migration discovery and Voice upgrade SQL."""

from scripts.apply_migrations import migration_files


def test_voice_migration_is_versioned_and_backfills_provenance():
    paths = migration_files()
    assert [path.name for path in paths] == [
        "20260820_voice_extractor_provenance.sql"
    ]
    sql = paths[0].read_text(encoding="utf-8")
    assert "candidate_extractor" in sql
    assert "provider_artifacts" in sql
    assert "extractor_provenance" in sql
    assert "WHERE idempotency_key IS NOT NULL" in sql
    assert "ALTER COLUMN extractor_provenance SET NOT NULL" in sql
