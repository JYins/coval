ALTER TABLE voice_ingestion_jobs
ADD COLUMN IF NOT EXISTS candidate_extractor VARCHAR(50);

UPDATE voice_ingestion_jobs
SET candidate_extractor = 'fake'
WHERE candidate_extractor IS NULL;

ALTER TABLE voice_ingestion_jobs
ALTER COLUMN candidate_extractor SET DEFAULT 'fake';

ALTER TABLE voice_ingestion_jobs
ALTER COLUMN candidate_extractor SET NOT NULL;

ALTER TABLE voice_ingestion_jobs
ADD COLUMN IF NOT EXISTS provider_artifacts JSONB;

UPDATE voice_ingestion_jobs
SET provider_artifacts = '{}'::jsonb
WHERE provider_artifacts IS NULL;

ALTER TABLE voice_ingestion_jobs
ALTER COLUMN provider_artifacts SET DEFAULT '{}'::jsonb;

ALTER TABLE voice_ingestion_jobs
ALTER COLUMN provider_artifacts SET NOT NULL;

ALTER TABLE extracted_candidates
ADD COLUMN IF NOT EXISTS extractor_provenance JSONB;

ALTER TABLE extracted_candidates
ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(255);

ALTER TABLE extracted_candidates
ADD COLUMN IF NOT EXISTS request_fingerprint VARCHAR(64);

UPDATE extracted_candidates
SET extractor_provenance = jsonb_build_object(
    'extractor_name', 'fake',
    'extractor_version', '1',
    'model_name', 'synthetic-candidate-fixture',
    'model_revision', 'legacy-backfill'
)
WHERE extractor_provenance IS NULL;

ALTER TABLE extracted_candidates
ALTER COLUMN extractor_provenance SET DEFAULT '{}'::jsonb;

ALTER TABLE extracted_candidates
ALTER COLUMN extractor_provenance SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_job_key
ON extracted_candidates (job_id, idempotency_key)
WHERE idempotency_key IS NOT NULL;

ALTER TABLE approved_memory_events
ADD COLUMN IF NOT EXISTS extractor_provenance JSONB;

ALTER TABLE approved_memory_events
ADD COLUMN IF NOT EXISTS provider_artifacts JSONB;

UPDATE approved_memory_events AS event
SET provider_artifacts = job.provider_artifacts
FROM voice_ingestion_jobs AS job
WHERE event.job_id = job.id
  AND event.provider_artifacts IS NULL;

UPDATE approved_memory_events
SET provider_artifacts = '{}'::jsonb
WHERE provider_artifacts IS NULL;

ALTER TABLE approved_memory_events
ALTER COLUMN provider_artifacts SET DEFAULT '{}'::jsonb;

ALTER TABLE approved_memory_events
ALTER COLUMN provider_artifacts SET NOT NULL;

UPDATE approved_memory_events AS event
SET extractor_provenance = candidate.extractor_provenance
FROM extracted_candidates AS candidate
WHERE event.candidate_id = candidate.id
  AND event.extractor_provenance IS NULL;

UPDATE approved_memory_events
SET extractor_provenance = '{}'::jsonb
WHERE extractor_provenance IS NULL;

ALTER TABLE approved_memory_events
ALTER COLUMN extractor_provenance SET DEFAULT '{}'::jsonb;

ALTER TABLE approved_memory_events
ALTER COLUMN extractor_provenance SET NOT NULL;
