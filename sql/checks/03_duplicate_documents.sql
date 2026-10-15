-- check: duplicate_documents
-- description: The same file received more than once against the same target. Both copies get read, and a field extracted from the stale copy silently wins if the reader sorts wrong.
SELECT
    d.entity_id,
    d.sha256,
    COUNT(*)                 AS copies,
    GROUP_CONCAT(d.id, ', ') AS document_ids,
    MIN(d.received_at)       AS first_seen,
    MAX(d.received_at)       AS last_seen
FROM documents AS d
WHERE d.entity_id IS NOT NULL
GROUP BY d.entity_id, d.sha256
HAVING COUNT(*) > 1
ORDER BY copies DESC, d.entity_id, d.sha256;
